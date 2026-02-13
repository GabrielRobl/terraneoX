

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/linearforms/shell/supg_rhs.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/pbicgstab.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using ScalarType = double;

/// Simple manufactured benchmark for the advection-diffusion equation.
///
/// Eq.
///
///   d/dt T + u · ∇T - κ ∇^2 T = H
///
/// Solution
///
///   T(t, x, y, z) = e^{λt} (x cos(Ωt) + y sin(Ωt))
///
///   H(t, x, y, z) = λT
///
///   u(t, x, y, z) = (-Ωy, Ωx, 0)
///
struct VelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_;
    ScalarType                     omega_;

    VelocityInterpolator(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataVec< ScalarType, 3 >& data,
        const ScalarType&                     omega )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , omega_( omega )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        data_( local_subdomain_id, x, y, r, 0 ) = -coords( 1 ) * omega_;
        data_( local_subdomain_id, x, y, r, 1 ) = coords( 0 ) * omega_;
        data_( local_subdomain_id, x, y, r, 2 ) = 0.0;
    }
};

struct SolutionAndRHSInterpolator
{
    Grid3DDataVec< ScalarType, 3 >                     grid_;
    Grid2DDataScalar< ScalarType >                     radii_;
    Grid4DDataScalar< ScalarType >                     data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;

    ScalarType t_;
    ScalarType lambda_;
    ScalarType omega_;

    bool is_rhs_;
    bool only_boundary_;

    SolutionAndRHSInterpolator(
        const Grid3DDataVec< ScalarType, 3 >&                     grid,
        const Grid2DDataScalar< ScalarType >&                     radii,
        const Grid4DDataScalar< ScalarType >&                     data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const ScalarType&                                         t,
        const ScalarType&                                         lambda,
        const ScalarType&                                         omega,
        const bool                                                is_rhs,
        const bool                                                only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , boundary_mask_( boundary_mask )
    , t_( t )
    , lambda_( lambda )
    , omega_( omega )
    , is_rhs_( is_rhs )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x_idx, const int y_idx, const int r_idx ) const
    {
        const dense::Vec< ScalarType, 3 > coords =
            grid::shell::coords( local_subdomain_id, x_idx, y_idx, r_idx, grid_, radii_ );

        const auto x = coords( 0 );
        const auto y = coords( 1 );

        if ( !only_boundary_ ||
             util::has_flag(
                 boundary_mask_( local_subdomain_id, x_idx, y_idx, r_idx ), grid::shell::ShellBoundaryFlag::BOUNDARY ) )
        {
            data_( local_subdomain_id, x_idx, y_idx, r_idx ) =
                Kokkos::exp( t_ * lambda_ ) * ( x * Kokkos::cos( omega_ * t_ ) + y * Kokkos::sin( omega_ * t_ ) );

            if ( is_rhs_ )
            {
                data_( local_subdomain_id, x_idx, y_idx, r_idx ) *= lambda_;
            }
        }
    }
};

void test( int level, const std::shared_ptr< util::Table >& table )
{
    constexpr int timesteps = 10;
    constexpr int restart   = 10;

    constexpr auto lambda = 1.0;
    constexpr auto omega  = 1.0;
    constexpr auto kappa  = 1.0;

    constexpr auto xdmf = true;

    ScalarType t = 0.0;

    const auto domain = DistributedDomain::create_uniform(
        level,
        grid::shell::mapped_shell_radii( 0.5, 1.0, ( 1 << level ) + 1, grid::shell::make_tanh_boundary_cluster( 2.0 ) ),
        0,
        0 );

    const auto h_min = grid::shell::min_radial_h( domain.domain_info().radii() );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorQ1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorQ1Scalar< ScalarType > g( "g", domain, mask_data );
    VectorQ1Scalar< ScalarType > f_strong( "f_strong", domain, mask_data );
    VectorQ1Scalar< ScalarType > f( "f", domain, mask_data );
    VectorQ1Vec< ScalarType >    u( "u", domain, mask_data );
    VectorQ1Scalar< ScalarType > solution( "solution", domain, mask_data );
    VectorQ1Scalar< ScalarType > error( "error", domain, mask_data );

    std::vector< VectorQ1Scalar< ScalarType > > tmps;
    for ( int i = 0; i < 2 * restart + 4; ++i )
    {
        tmps.emplace_back( "tmpp", domain, mask_data );
    }

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    std::cout << "Number of dofs: " << num_dofs << std::endl;

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;

    AD A( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, kappa, 0, true );
    AD A_neumann( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, kappa, 0, false );
    AD A_diagonal( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, kappa, 0, false, true );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;

    Mass                                   M( domain, subdomain_shell_coords, subdomain_radii );
    fe::wedge::linearforms::shell::SUPGRHS supg_rhs(
        domain, subdomain_shell_coords, subdomain_radii, f_strong, u, kappa );

    // Set up solution data.
    Kokkos::parallel_for(
        "velocity interpolation",
        local_domain_md_range_policy_nodes( domain ),
        VelocityInterpolator( subdomain_shell_coords, subdomain_radii, u.grid_data(), omega ) );

    Kokkos::fence();

    // Set up the initial temperature.
    Kokkos::parallel_for(
        "initial temp interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionAndRHSInterpolator(
            subdomain_shell_coords,
            subdomain_radii,
            T.grid_data(),
            boundary_mask_data,
            t,
            lambda,
            omega,
            false,
            false ) );

    Kokkos::fence();

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionAndRHSInterpolator(
            subdomain_shell_coords,
            subdomain_radii,
            solution.grid_data(),
            boundary_mask_data,
            t,
            lambda,
            omega,
            false,
            false ) );

    Kokkos::fence();

    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionAndRHSInterpolator(
            subdomain_shell_coords,
            subdomain_radii,
            f_strong.grid_data(),
            boundary_mask_data,
            t,
            lambda,
            omega,
            true,
            false ) );

    Kokkos::fence();

    linalg::solvers::FGMRES< AD > solver(
        tmps,
        { .restart                     = restart,
          .relative_residual_tolerance = 1e-6,
          .absolute_residual_tolerance = 1e-12,
          .max_iterations              = 100 },
        table );

    io::XDMFOutput xdmf_output(
        "test_unsteady_advection_diffusion_supg_2_output", domain, subdomain_shell_coords, subdomain_radii );
    xdmf_output.add( T.grid_data() );
    xdmf_output.add( solution.grid_data() );
    xdmf_output.add( error.grid_data() );

    if ( xdmf )
    {
        xdmf_output.write();
    }

    util::logroot << "Timestep " << 0 << std::endl;
    util::logroot << "  dt =     " << "-" << std::endl;
    util::logroot << "  h =      " << h_min << std::endl;
    util::logroot << "  kappa =  " << kappa << std::endl;

    linalg::lincomb( error, { 1.0, -1.0 }, { solution, T } );
    const auto l2_error = linalg::norm_2_scaled( error, 1.0 / static_cast< ScalarType >( num_dofs ) );
    util::logroot << "L2 error: " << l2_error << std::endl;

    for ( int ts = 1; ts < timesteps; ++ts )
    {
        const auto max_vel = kernels::common::max_vector_magnitude( u.grid_data() );

        // Choose "suitable" small dt for accuracy - we have and implicit time-stepping scheme so we do not really need
        // a CFL in the classical sense. Still useful for time-step size restriction.
        const auto dt_advection = h_min / max_vel;
        // const auto dt_diffusion = ( h * h ) / prm.diffusivity;
        // const auto dt           = prm.pseudo_cfl * std::min( dt_advection, dt_diffusion );
        auto dt = 0.5 * dt_advection;

        A.dt()          = dt;
        A_neumann.dt()  = dt;
        A_diagonal.dt() = dt;

        util::logroot << "Timestep " << ts << std::endl;
        util::logroot << "  dt =     " << dt << std::endl;
        util::logroot << "  h =      " << h_min << std::endl;
        util::logroot << "  kappa =  " << kappa << std::endl;
        t += dt;

        Kokkos::parallel_for(
            "rhs interpolation",
            local_domain_md_range_policy_nodes( domain ),
            SolutionAndRHSInterpolator(
                subdomain_shell_coords,
                subdomain_radii,
                f_strong.grid_data(),
                boundary_mask_data,
                t,
                lambda,
                omega,
                true,
                false ) );

        assign( g, 0.0 );
        Kokkos::parallel_for(
            "boundary temp interpolation",
            local_domain_md_range_policy_nodes( domain ),
            SolutionAndRHSInterpolator(
                subdomain_shell_coords,
                subdomain_radii,
                g.grid_data(),
                boundary_mask_data,
                t,
                lambda,
                omega,
                false,
                true ) );

        Kokkos::fence();

        assign( tmps[0], 0.0 );
        assign( tmps[1], 0.0 );
        assign( tmps[2], 0.0 );

        linalg::apply( M, T, tmps[0] );
        linalg::apply( M, f_strong, tmps[1] );
        // linalg::apply( supg_rhs, tmps[2] );

        lincomb( f, { 1.0, dt, dt }, { tmps[0], tmps[1], tmps[2] } );

        assign( tmps[0], 0.0 );
        fe::strong_algebraic_dirichlet_enforcement_poisson_like(
            A_neumann, A_diagonal, g, tmps[0], f, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

        linalg::solvers::solve( solver, A, T, f );

        // table->print_pretty();
        table->clear();

        Kokkos::parallel_for(
            "solution interpolation",
            local_domain_md_range_policy_nodes( domain ),
            SolutionAndRHSInterpolator(
                subdomain_shell_coords,
                subdomain_radii,
                solution.grid_data(),
                boundary_mask_data,
                t,
                lambda,
                omega,
                false,
                false ) );

        Kokkos::fence();

        linalg::lincomb( error, { 1.0, -1.0 }, { solution, T } );
        const auto l2_error = linalg::norm_2_scaled( error, 1.0 / static_cast< ScalarType >( num_dofs ) );
        util::logroot << "L2 error: " << l2_error << std::endl;

        if ( xdmf )
        {
            xdmf_output.write();
        }
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    const int level = 4;

    test( level, table );

    return 0;
}