
#pragma once
#include "communication/shell/fv_communication.hpp"
#include "fe/triangle/quadrature/quadrature.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fv::hex::operators {

template < typename ScalarT >
class UnsteadyAdvectionDiffusion
{
  public:
    using SrcVectorType = linalg::VectorFVScalar< ScalarT >;
    using DstVectorType = linalg::VectorFVScalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    static constexpr int num_velocity_components = 3;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataVec< ScalarT, 3 >                        cell_centers_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;

    linalg::VectorQ1Vec< ScalarT, 3 > velocity_;

    ScalarT diffusivity_;
    ScalarT dt_;

    /// Pre-allocated send/recv buffers for ghost layer communication (built once in ctor).
    communication::shell::FVGhostLayerBuffers< ScalarType > ghost_bufs_;

    grid::Grid4DDataScalar< ScalarType >                      src_;
    grid::Grid4DDataScalar< ScalarType >                      dst_;
    grid::Grid4DDataVec< ScalarType, num_velocity_components > vel_grid_;

  public:
    UnsteadyAdvectionDiffusion(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataVec< ScalarT, 3 >&                        cell_centers,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const linalg::VectorQ1Vec< ScalarT, num_velocity_components >&   velocity,
        const ScalarT                                                   diffusivity,
        const ScalarT                                                   dt )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , cell_centers_( cell_centers )
    , boundary_mask_( boundary_mask )
    , velocity_( velocity )
    , diffusivity_( diffusivity )
    , dt_( dt )
    , ghost_bufs_( domain )
    {}

    ScalarT&       dt() { return dt_; }
    const ScalarT& dt() const { return dt_; }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        // Fill ghost layers so that stencil reads at ±1 offsets see up-to-date neighbour values.
        // Uses pre-allocated buffers — no heap allocation on this hot path.
        communication::shell::update_fv_ghost_layers( domain_, src.grid_data(), ghost_bufs_ );

        src_      = src.grid_data();
        dst_      = dst.grid_data();
        vel_grid_ = velocity_.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            Kokkos::abort( "src and dst extents do not match" );
        }

        if ( src_.extent( 0 ) != cell_centers_.extent( 0 ) || src_.extent( 1 ) != cell_centers_.extent( 1 ) ||
             src_.extent( 2 ) != cell_centers_.extent( 2 ) || src_.extent( 3 ) != cell_centers_.extent( 3 ) )
        {
            Kokkos::abort( "src and dst extents do not match" );
        }

        Kokkos::parallel_for(
            "matvec",
            Kokkos::MDRangePolicy(
                { 0, 1, 1, 1 },
                { src.grid_data().extent( 0 ),
                  src.grid_data().extent( 1 ) - 1,
                  src.grid_data().extent( 2 ) - 1,
                  src.grid_data().extent( 3 ) - 1 } ),
            *this );

        Kokkos::fence();
    }

  private:
    using Vec3 = dense::Vec< ScalarType, 3 >;

    static constexpr int num_neighbors                = 6;
    static constexpr int cell_offset_x[num_neighbors] = { -1, 1, 0, 0, 0, 0 };
    static constexpr int cell_offset_y[num_neighbors] = { 0, 0, -1, 1, 0, 0 };
    static constexpr int cell_offset_r[num_neighbors] = { 0, 0, 0, 0, -1, 1 };

    enum class WedgeContribution : int
    {
        NONE     = 0,
        TRIANGLE = 1,
        QUAD     = 2,
    };

    enum class CrossProductType : int
    {
        XI_ETA   = 0,
        XI_ZETA  = 1,
        ETA_ZETA = 2
    };

    KOKKOS_INLINE_FUNCTION
    static constexpr WedgeContribution contribution( const int hex_direction, const int wedge_id )
    {
        if ( wedge_id == 0 )
        {
            switch ( hex_direction )
            {
            case 0:
                // hex face left
                return WedgeContribution::QUAD;

            case 2:
                // hex face front
                return WedgeContribution::QUAD;

            case 4:
                // hex face bottom
                return WedgeContribution::TRIANGLE;

            case 5:
                // hex face top
                return WedgeContribution::TRIANGLE;

            default:
                return WedgeContribution::NONE;
            }
        }
        else
        {
            switch ( hex_direction )
            {
            case 1:
                // hex face right
                return WedgeContribution::QUAD;

            case 3:
                // hex face back
                return WedgeContribution::QUAD;

            case 4:
                // hex face bottom
                return WedgeContribution::TRIANGLE;

            case 5:
                // hex face top
                return WedgeContribution::TRIANGLE;

            default:
                return WedgeContribution::NONE;
            }
        }
    }

    /// Map reference face coordinates (s,t) to (xi,eta,zeta) in wedge
    KOKKOS_INLINE_FUNCTION
    static constexpr void map_face(
        const int         hex_direction,
        const int         wedge_id,
        const double      s,
        const double      t,
        double&           xi,
        double&           eta,
        double&           zeta,
        CrossProductType& cross_product_type )
    {
        if ( wedge_id == 0 )
        {
            switch ( hex_direction )
            {
            case 0:
                xi                 = 0;
                eta                = s;
                zeta               = t;
                cross_product_type = CrossProductType::ETA_ZETA;
                break; // hex face left
            case 2:
                xi                 = s;
                eta                = 0;
                zeta               = t;
                cross_product_type = CrossProductType::XI_ZETA;
                break; // hex face front
            case 4:
                xi                 = s;
                eta                = t;
                zeta               = -1;
                cross_product_type = CrossProductType::XI_ETA;
                break; // hex face bottom
            case 5:
                xi                 = s;
                eta                = t;
                zeta               = 1;
                cross_product_type = CrossProductType::XI_ETA;
                break; // hex face top
            default:
                xi                 = 0;
                eta                = 0;
                zeta               = 0;
                cross_product_type = CrossProductType::XI_ETA;
                break; // this should not happen
            }
        }
        else
        {
            switch ( hex_direction )
            {
            case 1:
                xi                 = 0;
                eta                = s;
                zeta               = t;
                cross_product_type = CrossProductType::ETA_ZETA;
                break; // hex face right
            case 3:
                xi                 = s;
                eta                = 0;
                zeta               = t;
                cross_product_type = CrossProductType::XI_ZETA;
                break; // hex face back
            case 4:
                xi                 = s;
                eta                = t;
                zeta               = -1;
                cross_product_type = CrossProductType::XI_ETA;
                break; // hex face bottom
            case 5:
                xi                 = s;
                eta                = t;
                zeta               = 1;
                cross_product_type = CrossProductType::XI_ETA;
                break; // hex face top
            default:
                xi                 = 0;
                eta                = 0;
                zeta               = 0;
                cross_product_type = CrossProductType::XI_ETA;
                break; // this should not happen
            }
        }
    }

  public:
    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Gather surface points for each wedge.
        dense::Vec< ScalarT, 3 > wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                               [fe::wedge::num_nodes_per_wedge_surface] = {};
        fe::wedge::wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell - 1, y_cell - 1 );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell - 1 );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell );

        // Quadrature points.
        constexpr auto num_quad_points = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points );
        fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        // Velocity coefficients.

        dense::Vec< ScalarT, 6 > vel_coeffs[num_velocity_components][fe::wedge::num_wedges_per_hex_cell];

        for ( int d = 0; d < num_velocity_components; d++ )
        {
            fe::wedge::extract_local_wedge_vector_coefficients(
                vel_coeffs[d], local_subdomain_id, x_cell - 1, y_cell - 1, r_cell - 1, d, vel_grid_ );
        }

        // Compute vector surface integral and flux for all directions
        // we are computing two terms at once
        //
        // for each hex face f
        //
        //      S_f = sum_wedges int_f_wedge n dS    (vector surface integral, vector-valued, later required for diffusion)
        // and
        //
        //      beta = sum_wedges int_f_wedge u.n dS    (flux, scalar, later required for advection)
        //
        // For that, we use a parameterization of the faces via local coordinates (s, t):
        //
        //      x(s, t) = x( xi(s, t), eta(s, t), zeta(s, t) )
        //
        // where the functions xi(s, t), eta(s, t), zeta(s, t) depend on the respective facet.
        // The functions are evaluated at (s, t) via map_face() (returning xi, eta, zeta).
        //
        // From differential geometry we have
        //
        //      int n dS = int cross(dx/ds, dx/dt) ds dt
        //
        // and
        //
        //      int u.n dS = int u(x(s, t)) . cross(dx/ds, dx/dt) ds dt
        //
        // Note that from the chain rule we get
        //
        //      dx/ds = dx/dxi * dxi/ds + dx/deta * deta/ds + dx/dzeta * dzeta/ds
        //
        // and corresponding terms for dx/dt.
        //
        // The mappings x(s, t) are typically so simple that the derivatives are either 1 or 0:
        //
        //      example: left hex facet == wedge 0 with xi = 0
        //      then: xi(s, t) = 0, eta(s, t) = s, zeta(s, t) = t
        //      and we have
        //
        //      dx/ds = dx/dxi * dxi/ds + dx/deta * deta/ds + dx/dzeta * dzeta/ds
        //            = dx/dxi *      0 + dx/deta *       1 + dx/dzeta *        0
        //            = dx/deta
        //
        // Luckily this works for all the sides, so we only ever compute dx/dxi, dx/deta, and dx/dzeta, which are just
        // the shape function gradients of the wedge. Just need to make sure to use the right terms for each facet.
        //
        // We really only have the functionality to integrate over wedges (and their facets), but no "general" way (yet)
        // to integrate over the hex facets directly. So we build the hex facet integrals from wedge facet integrals.
        // Since each hex is split into two wedges, the details of the integration over the hex facets depends
        // on the side of the hex. At the top and bottom we execute two integrals, one over the triangular surface of
        // ether wedge.
        // On the other sides, we only have contributions from the quad sides of a single wedge.

        Vec3       S_f[num_neighbors]  = {};
        ScalarType beta[num_neighbors] = {};

        Vec3 cell_center{
            cell_centers_( local_subdomain_id, x_cell, y_cell, r_cell, 0 ),
            cell_centers_( local_subdomain_id, x_cell, y_cell, r_cell, 1 ),
            cell_centers_( local_subdomain_id, x_cell, y_cell, r_cell, 2 ) };

        Vec3 neighbor_cell_centers[num_neighbors] = {};

        for ( int neighbor = 0; neighbor < num_neighbors; ++neighbor )
        {
            for ( int d = 0; d < 3; ++d )
            {
                neighbor_cell_centers[neighbor]( d ) = cell_centers_(
                    local_subdomain_id,
                    x_cell + cell_offset_x[neighbor],
                    y_cell + cell_offset_y[neighbor],
                    r_cell + cell_offset_r[neighbor],
                    d );
            }
        }

        constexpr auto num_quad_points_tri = fe::triangle::quadrature::quad_triangle_3_num_quad_points;

        dense::Vec< ScalarT, 2 > quad_points_tri[num_quad_points_tri];
        ScalarT                  quad_weights_tri[num_quad_points_tri];

        fe::triangle::quadrature::quad_triangle_3_quad_points( quad_points_tri );
        fe::triangle::quadrature::quad_triangle_3_quad_weights( quad_weights_tri );

        constexpr auto num_quad_points_line                    = 2;
        ScalarT        quad_points_line[num_quad_points_line]  = { -1.0 / Kokkos::sqrt( 3 ), 1.0 / Kokkos::sqrt( 3 ) };
        ScalarT        quad_weights_line[num_quad_points_line] = { 1.0, 1.0 };

        for ( int neighbor = 0; neighbor < num_neighbors; ++neighbor )
        {
            for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
            {
                const auto contr = contribution( neighbor, wedge );

                if ( contr == WedgeContribution::NONE )
                {
                    continue;
                }

                if ( contr == WedgeContribution::TRIANGLE )
                {
                    for ( int q = 0; q < num_quad_points_tri; ++q )
                    {
                        const auto s = quad_points_tri[q]( 0 );
                        const auto t = quad_points_tri[q]( 1 );

                        ScalarType       xi   = 0;
                        ScalarType       eta  = 0;
                        ScalarType       zeta = 0;
                        CrossProductType cross_product_type;

                        map_face( neighbor, wedge, s, t, xi, eta, zeta, cross_product_type );

                        Vec3 dx_dxi   = {};
                        Vec3 dx_deta  = {};
                        Vec3 dx_dzeta = {};
                        Vec3 u        = {};

                        for ( int i = 0; i < fe::wedge::num_nodes_per_wedge; ++i )
                        {
                            const auto shape_i = fe::wedge::shape( i, xi, eta, zeta );
                            const auto grad_i  = fe::wedge::grad_shape( i, xi, eta, zeta );

                            const Vec3 x_i = wedge_phy_surf[wedge][i % 3] * ( i < 3 ? r_1 : r_2 );
                            const Vec3 u_i{
                                vel_coeffs[0][wedge]( i ), vel_coeffs[1][wedge]( i ), vel_coeffs[2][wedge]( i ) };

                            dx_dxi   = dx_dxi + grad_i( 0 ) * x_i;
                            dx_deta  = dx_deta + grad_i( 1 ) * x_i;
                            dx_dzeta = dx_dzeta + grad_i( 2 ) * x_i;

                            u = u + shape_i * u_i;
                        }

                        Vec3 n = {};
                        if ( cross_product_type == CrossProductType::XI_ETA )
                        {
                            n = dx_dxi.cross( dx_deta );
                        }
                        else
                        {
                            Kokkos::abort( "This should not happen on triangular surfaces." );
                        }

                        beta[neighbor] += quad_weights_tri[q] * u.dot( n );
                        S_f[neighbor] = S_f[neighbor] + quad_weights_tri[q] * n;
                    }
                }

                if ( contr == WedgeContribution::QUAD )
                {
                    for ( int q_a = 0; q_a < num_quad_points_line; ++q_a )
                    {
                        for ( int q_b = 0; q_b < num_quad_points_line; ++q_b )
                        {
                            const auto s = quad_points_line[q_a];
                            const auto t = quad_points_line[q_b];
                            const auto w = quad_weights_line[q_a] * quad_weights_line[q_b];

                            ScalarType       xi   = 0;
                            ScalarType       eta  = 0;
                            ScalarType       zeta = 0;
                            CrossProductType cross_product_type;

                            map_face( neighbor, wedge, s, t, xi, eta, zeta, cross_product_type );

                            Vec3 dx_dxi   = {};
                            Vec3 dx_deta  = {};
                            Vec3 dx_dzeta = {};
                            Vec3 u        = {};

                            for ( int i = 0; i < fe::wedge::num_nodes_per_wedge; ++i )
                            {
                                const auto shape_i = fe::wedge::shape( i, xi, eta, zeta );
                                const auto grad_i  = fe::wedge::grad_shape( i, xi, eta, zeta );

                                const Vec3 x_i = wedge_phy_surf[wedge][i % 3] * ( i < 3 ? r_1 : r_2 );
                                const Vec3 u_i{
                                    vel_coeffs[0][wedge]( i ), vel_coeffs[1][wedge]( i ), vel_coeffs[2][wedge]( i ) };

                                dx_dxi   = dx_dxi + grad_i( 0 ) * x_i;
                                dx_deta  = dx_deta + grad_i( 1 ) * x_i;
                                dx_dzeta = dx_dzeta + grad_i( 2 ) * x_i;

                                u = u + shape_i * u_i;
                            }

                            Vec3 n = {};

                            if ( cross_product_type == CrossProductType::XI_ZETA )
                            {
                                n = dx_dxi.cross( dx_dzeta );
                            }
                            else if ( cross_product_type == CrossProductType::ETA_ZETA )
                            {
                                n = dx_deta.cross( dx_dzeta );
                            }
                            else
                            {
                                Kokkos::abort( "This should not happen on quad surfaces." );
                            }

                            beta[neighbor] += w * u.dot( n );
                            S_f[neighbor] = S_f[neighbor] + w * n;
                        }
                    }
                }
            }

            if ( S_f[neighbor].dot( neighbor_cell_centers[neighbor] - cell_center ) < 0 )
            {
                S_f[neighbor] = S_f[neighbor] * ( -1.0 );
                beta[neighbor] *= -1.0;
            }
        }



        // Cell volume (for mass matrix)
        ScalarType M_ii = 0;
        for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
        {
            for ( int q = 0; q < num_quad_points; ++q )
            {
                const auto J       = fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det     = J.det();
                const auto abs_det = Kokkos::abs( det );
                const auto Jw      = abs_det * quad_weights[q];
                M_ii += Jw;
            }
        }

        // Assemble stencil

        ScalarType AD_ii                = 0;
        ScalarType AD_ij[num_neighbors] = {};

        for ( int neighbor = 0; neighbor < num_neighbors; ++neighbor )
        {
            const auto S_f_n = S_f[neighbor];

            // diffusion

            const Vec3       dx                = neighbor_cell_centers[neighbor] - cell_center;
            const ScalarType denom             = dx.dot( S_f_n );
            const ScalarType offdiag_diffusion = -diffusivity_ * ( S_f_n.dot( S_f_n ) / denom );
            const ScalarType diag_diffusion    = -offdiag_diffusion;

            // advection
            ScalarType offdiag_advection = -beta[neighbor];
            ScalarType diag_advection    = 0;
            if ( beta[neighbor] >= 0 )
            {
                offdiag_advection = 0;
                diag_advection    = beta[neighbor];
            }

            // TODO: if dirichlet(neighbor) then offdiag[neighbor] = 0
            AD_ii += diag_advection + diag_diffusion;
            AD_ij[neighbor] = offdiag_diffusion + offdiag_advection;
        }

        // Apply stencil.
        // Mass is diagonal.

        dst_( local_subdomain_id, x_cell, y_cell, r_cell ) =
            ( M_ii + dt_ * AD_ii ) * src_( local_subdomain_id, x_cell, y_cell, r_cell );

        for ( int neighbor = 0; neighbor < num_neighbors; ++neighbor )
        {
            dst_( local_subdomain_id, x_cell, y_cell, r_cell ) += dt_ * AD_ij[neighbor] *
                                                                  src_(
                                                                      local_subdomain_id,
                                                                      x_cell + cell_offset_x[neighbor],
                                                                      y_cell + cell_offset_y[neighbor],
                                                                      r_cell + cell_offset_r[neighbor] );
        }
    }
};

} // namespace terra::fv::hex::operators