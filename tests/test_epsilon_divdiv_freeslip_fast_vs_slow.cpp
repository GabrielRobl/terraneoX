#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::BoundaryConditions;
using grid::shell::DistributedDomain;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

struct VectorFieldInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    VectorFieldInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const auto rr     = radii_( local_subdomain_id, r );

        data_( local_subdomain_id, x, y, r, 0 ) =
            ( rr - 1.0 ) * ( rr - 0.5 ) * 0.5 * Kokkos::sin( 2.0 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
        data_( local_subdomain_id, x, y, r, 1 ) =
            ( rr - 1.0 ) * ( rr - 0.5 ) * 0.5 * Kokkos::sin( 3.0 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
        data_( local_subdomain_id, x, y, r, 2 ) =
            ( rr - 1.0 ) * ( rr - 0.5 ) * 0.5 * Kokkos::sin( 4.0 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
    }
};

struct ScalarCoeffInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    ScalarCoeffInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const auto rr     = radii_( local_subdomain_id, r );

        // Smooth positive coefficient.
        data_( local_subdomain_id, x, y, r ) =
            1.0 + 0.1 * ( rr - 0.75 ) + 0.05 * Kokkos::cos( coords( 0 ) ) * Kokkos::cosh( 0.25 * coords( 1 ) );
    }
};

template < typename ScalarT >
void compare_epsilon_divdiv_freeslip_cmb_dirichlet_surface( int level, bool diagonal )
{
    using Op = fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarT >;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarT >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarT >( domain );

    VectorQ1Vec< ScalarT > src( "src", domain, mask_data );
    VectorQ1Vec< ScalarT > dst_fast( "dst_fast", domain, mask_data );
    VectorQ1Vec< ScalarT > dst_slow( "dst_slow", domain, mask_data );
    VectorQ1Vec< ScalarT > err( "err", domain, mask_data );

    VectorQ1Scalar< ScalarT > k_coeff( "k", domain, mask_data );
    VectorQ1Scalar< ScalarT > gca_elements( "gca_elements", domain, mask_data );

    Kokkos::parallel_for(
        "interpolate src",
        local_domain_md_range_policy_nodes( domain ),
        VectorFieldInterpolator( coords_shell, coords_radii, src.grid_data() ) );

    Kokkos::parallel_for(
        "interpolate k",
        local_domain_md_range_policy_nodes( domain ),
        ScalarCoeffInterpolator( coords_shell, coords_radii, k_coeff.grid_data() ) );

    // gca_elements is only a placeholder here; keep zero.
    assign( gca_elements, 0 );

    Kokkos::fence();

    BoundaryConditions bcs = {
        { CMB, FREESLIP },     // <- requested
        { SURFACE, DIRICHLET } // <- requested
    };

    // Fast operator: matrix-free, no stored matrices -> should dispatch to fast_freeslip.
    Op op_fast(
        domain,
        coords_shell,
        coords_radii,
        boundary_mask_data,
        k_coeff.grid_data(),
        bcs,
        diagonal,
        linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode::Off );

    // Slow operator: force local-matrix path by enabling selective stored-matrix mode.
    Op op_slow(
        domain,
        coords_shell,
        coords_radii,
        boundary_mask_data,
        k_coeff.grid_data(),
        bcs,
        diagonal,
        linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode::Off );

    op_slow.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Selective, /*level_range=*/0, gca_elements.grid_data() );

    linalg::apply( op_fast, src, dst_fast );
    linalg::apply( op_slow, src, dst_slow );

    linalg::lincomb( err, { 1.0, -1.0 }, { dst_fast, dst_slow } );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    const auto l2_err   = std::sqrt( dot( err, err ) / num_dofs );
    const auto inf_err  = linalg::norm_inf( err );

    std::cout << "  L2 error  = " << l2_err << std::endl;
    std::cout << "  inf error = " << inf_err << std::endl;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    for ( auto diagonal : { true, false } )
    {
        std::cout << "==================================================" << std::endl;
        std::cout << "EpsilonDivDivKerngen fast_freeslip vs slow" << std::endl;
        std::cout << "BCs: CMB=FREESLIP, SURFACE=DIRICHLET" << std::endl;
        std::cout << "diagonal = " << diagonal << std::endl;

        for ( int level = 0; level < 6; ++level )
        {
            std::cout << "level = " << level << std::endl;
            compare_epsilon_divdiv_freeslip_cmb_dirichlet_surface< double >( level, diagonal );
        }
    }

    return 0;
}