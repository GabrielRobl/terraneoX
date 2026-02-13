#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "impl/Kokkos_Profiling.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers/gca/local_matrix_storage.hpp"
#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell {

using grid::shell::get_boundary_condition_flag;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using terra::grid::shell::BoundaryConditionFlag;
using terra::grid::shell::BoundaryConditions;
using terra::grid::shell::ShellBoundaryFlag;
using terra::linalg::trafo::trafo_mat_cartesian_to_normal_tangential;

template < typename ScalarT, int VecDim = 3 >
class EpsilonDivDivKerngen
{
  public:
    using SrcVectorType                 = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType                 = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType                    = ScalarT;
    static constexpr int LocalMatrixDim = 18;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, LocalMatrixDim, LocalMatrixDim, 2 >;
    using LocalMatrixStorage      = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >;

  private:
    LocalMatrixStorage local_matrix_storage_;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< ScalarType >                     k_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    BoundaryConditions                                       bcs_;
    BoundaryConditions                                       bcs_;

    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;
    linalg::OperatorStoredMatrixMode  operator_stored_matrix_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

    // Quadrature points.
    const int num_quad_points = quadrature::quad_felippa_1x1_num_quad_points;

    dense::Vec< ScalarT, 3 > quad_points[quadrature::quad_felippa_1x1_num_quad_points];
    ScalarT                  quad_weights[quadrature::quad_felippa_1x1_num_quad_points];

    int local_subdomains_;
    int hex_lat_;
    int hex_rad_;
    int lat_refinement_level_;
    int block_size_;
    int blocks_per_column_;
    int blocks_;

    ScalarT r_max_;
    ScalarT r_min_;

  public:
    EpsilonDivDivKerngen(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const grid::Grid4DDataScalar< ScalarT >&                        k,
        BoundaryConditions                                              bcs,
        bool                              diagonal,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode operator_stored_matrix_mode = linalg::OperatorStoredMatrixMode::Off )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , mask_( mask )
    , k_( k )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , operator_stored_matrix_mode_( operator_stored_matrix_mode )
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];
        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );
        const grid::shell::DomainInfo& domain_info = domain_.domain_info();
        local_subdomains_                          = domain_.subdomains().size();
        hex_lat_                                   = domain_info.subdomain_num_nodes_per_side_laterally() - 1;
        hex_rad_                                   = domain_info.subdomain_num_nodes_radially() - 1;
        lat_refinement_level_                      = domain_info.diamond_lateral_refinement_level();
        const int threads_per_column               = hex_rad_;
        block_size_                                = std::min( 256, threads_per_column );
        blocks_per_column_                         = ( threads_per_column + block_size_ - 1 ) / block_size_;
        blocks_                                    = local_subdomains_ * hex_lat_ * hex_lat_ * blocks_per_column_;
        r_min_                                     = domain_info.radii()[0];
        r_max_                                     = domain_info.radii()[domain_info.radii().size() - 1];
        util::logroot << "[EpsilonDivDiv] (threads_per_column, block_size_, blocks_per_column_) = "
                      << threads_per_column << ", " << block_size_ << ", " << blocks_per_column_ << ")" << std::endl;
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    /// @brief S/Getter for diagonal member
    void set_diagonal( bool v ) { diagonal_ = v; }

    /// @brief Getter for coefficient
    const grid::Grid4DDataScalar< ScalarType >& k_grid_data() { return k_; }

    /// @brief Getter for domain member
    const grid::shell::DistributedDomain& get_domain() const { return domain_; }

    /// @brief Getter for radii member
    grid::Grid2DDataScalar< ScalarT > get_radii() const { return radii_; }

    /// @brief Getter for grid member
    grid::Grid3DDataVec< ScalarT, 3 > get_grid() { return grid_; }

    /// @brief Getter for mask member
    KOKKOS_INLINE_FUNCTION
    bool has_flag(
        const int                      local_subdomain_id,
        const int                      x_cell,
        const int                      y_cell,
        const int                      r_cell,
        grid::shell::ShellBoundaryFlag flag ) const
    {
        return util::has_flag( mask_( local_subdomain_id, x_cell, y_cell, r_cell ), flag );
    }

    /// @brief allocates memory for the local matrices
    void set_stored_matrix_mode(
        linalg::OperatorStoredMatrixMode     operator_stored_matrix_mode,
        int                                  level_range,
        grid::Grid4DDataScalar< ScalarType > GCAElements )
    {
        operator_stored_matrix_mode_ = operator_stored_matrix_mode;

        // allocate storage if necessary
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            local_matrix_storage_ = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >(
                domain_, operator_stored_matrix_mode_, level_range, GCAElements );
        }
    }

    linalg::OperatorStoredMatrixMode get_stored_matrix_mode() { return operator_stored_matrix_mode_; }

    /// @brief Set the local matrix stored in the operator
    KOKKOS_INLINE_FUNCTION
    void set_local_matrix(
        const int                                                    local_subdomain_id,
        const int                                                    x_cell,
        const int                                                    y_cell,
        const int                                                    r_cell,
        const int                                                    wedge,
        const dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim >& mat ) const
    {
        KOKKOS_ASSERT( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off );
        local_matrix_storage_.set_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge, mat );
    }

    /// @brief Retrives the local matrix
    /// if there is stored local matrices, the desired local matrix is loaded and returned
    /// if not, the local matrix is assembled on-the-fly
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > get_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        // request from storage
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            if ( !local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) )
            {
                Kokkos::abort( "No matrix found at that spatial index." );
            }
            return local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
        }
        else
        {
            return assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
        }
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "epsilon_divdiv_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            throw std::runtime_error( "EpsilonDivDiv: src/dst mismatch" );
        }

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "EpsilonDivDiv: src/dst mismatch" );
        }

        util::Timer          timer_kernel( "epsilon_divdiv_kernel" );
        Kokkos::TeamPolicy<> policy( blocks_, block_size_ );
        Kokkos::parallel_for( "matvec", policy, *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "epsilon_divdiv_comm" );

            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION
    void column_grad_to_sym(
        const int    dim,
        const double g0,
        const double g1,
        const double g2,
        double&      E00,
        double&      E11,
        double&      E22,
        double&      sym01,
        double&      sym02,
        double&      sym12,
        double&      gdd ) const
    {
        E00   = 0.0;
        E11   = 0.0;
        E22   = 0.0;
        sym01 = 0.0;
        sym02 = 0.0;
        sym12 = 0.0;
        gdd   = 0.0;

        // dim selects which COLUMN is populated:
        // dim==0: E[0][0]=g0, E[1][0]=g1, E[2][0]=g2
        // dim==1: E[0][1]=g0, E[1][1]=g1, E[2][1]=g2
        // dim==2: E[0][2]=g0, E[1][2]=g1, E[2][2]=g2
        switch ( dim )
        {
        case 0:
            E00   = g0;
            gdd   = g0;       // E[0][0]
            sym01 = 0.5 * g1; // 0.5*(E[0][1]=0 + E[1][0]=g1)
            sym02 = 0.5 * g2; // 0.5*(E[0][2]=0 + E[2][0]=g2)
            sym12 = 0.0;
            break;

        case 1:
            E11   = g1;
            gdd   = g1;       // E[1][1]
            sym01 = 0.5 * g0; // 0.5*(E[0][1]=g0 + E[1][0]=0)
            sym02 = 0.0;
            sym12 = 0.5 * g2; // 0.5*(E[1][2]=0 + E[2][1]=g2)
            break;

        default: // case 2
            E22   = g2;
            gdd   = g2; // E[2][2]
            sym01 = 0.0;
            sym02 = 0.5 * g0; // 0.5*(E[0][2]=g0 + E[2][0]=0)
            sym12 = 0.5 * g1; // 0.5*(E[1][2]=g1 + E[2][1]=0)
            break;
        }
    }

    KOKKOS_INLINE_FUNCTION
    static size_t team_shmem_size( const int ts /*team_size*/ )
    {
        // We store wedge_surf_phy_coords[2][3][3], src, k
        return sizeof( ScalarType ) * ( 2 * 3 * 3 + ( ts + 1 ) * ( 12 + 4 + 1 ) );
    }

    using Team = Kokkos::TeamPolicy<>::member_type;

    KOKKOS_INLINE_FUNCTION void operator()( const Team& team ) const
    {
        int local_subdomain_id, x_cell, y_cell, r_cell;

        int r_block_index = 0;
        {
            int tmp       = team.league_rank();
            r_block_index = tmp % blocks_per_column_;
            tmp /= blocks_per_column_;
            y_cell = tmp & ( hex_lat_ - 1 );
            tmp >>= lat_refinement_level_;
            x_cell = tmp & ( hex_lat_ - 1 );
            tmp >>= lat_refinement_level_;
            local_subdomain_id = tmp;

            r_cell = r_block_index * team.team_size() + team.team_rank();
        }

        const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
        const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off || at_cmb || at_surface )
        {
            dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A[num_wedges_per_hex_cell] = { 0 };

            if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Full )
            {
                // gca on all elements, load local matrices
                A[0] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
                A[1] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
            }
            else if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Selective )
            {
                // adaptive gca: check if we have a matrix stored for this element
                if ( local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 ) &&
                     local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 ) )
                {
                    // large coefficient variation element: use stored gca matrix
                    A[0] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
                    A[1] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
                }
                else
                {
                    // low coefficient variation: assemble matrix on-the-fly using dca
                    A[0] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
                    A[1] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
                }
            }
            else
            {
                // boundary element: assemble matrix on-the-fly using dca
                A[0] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
                A[1] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
            }

            // read source dofs
            dense::Vec< ScalarT, 18 > src[num_wedges_per_hex_cell];
            for ( int dimj = 0; dimj < 3; dimj++ )
            {
                dense::Vec< ScalarT, 6 > src_d[num_wedges_per_hex_cell];
                extract_local_wedge_vector_coefficients(
                    src_d, local_subdomain_id, x_cell, y_cell, r_cell, dimj, src_ );

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        src[wedge]( dimj * num_nodes_per_wedge + i ) = src_d[wedge]( i );
                    }
                }
            }

            // Boundary treatment
            dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > boundary_mask;
            boundary_mask.fill( 1.0 );

            // flag to later not go through the hustle of checking the bcs
            bool                                                  freeslip_reorder = false;
            dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > R[num_wedges_per_hex_cell];

            if ( at_cmb || at_surface )
            {
                // Inner boundary (CMB).
                ShellBoundaryFlag     sbf = at_cmb ? CMB : SURFACE;
                BoundaryConditionFlag bcf = get_boundary_condition_flag( bcs_, sbf );

                if ( bcf == DIRICHLET )
                {
                    for ( int dimi = 0; dimi < 3; ++dimi )
                    {
                        for ( int dimj = 0; dimj < 3; ++dimj )
                        {
                            for ( int i = 0; i < num_nodes_per_wedge; i++ )
                            {
                                for ( int j = 0; j < num_nodes_per_wedge; j++ )
                                {
                                    if ( ( at_cmb && ( ( dimi == dimj /* at diagonal eps component */ &&
                                                         i != j /* dont kill the diag */ &&
                                                         ( i < 3 || j < 3 /* CMB -> bc at bot layer dofs */ ) ) ||
                                                       ( dimi != dimj /* off-diagonal eps component, kill diag too */ &&
                                                         ( i < 3 || j < 3 ) ) ) ) ||
                                         ( at_surface &&
                                           ( ( dimi == dimj /* at diagonal eps component */ &&
                                               i != j /* dont kill the diag */ &&
                                               ( i >= 3 || j >= 3 /* SUFACE -> bc at top layer dofs */ ) ) ||
                                             ( dimi != dimj /* off-diagonal eps component, kill diag too */ &&
                                               ( i >= 3 || j >= 3 ) ) ) ) )
                                    {
                                        boundary_mask(
                                            i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
                else if ( bcf == FREESLIP )
                {
                    freeslip_reorder                                                                     = true;
                    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A_tmp[num_wedges_per_hex_cell] = { 0 };

                    // reorder source dofs for nodes instead of velocity dims in src vector and local matrix
                    for ( int wedge = 0; wedge < 2; ++wedge )
                    {
                        for ( int dimi = 0; dimi < 3; ++dimi )
                        {
                            for ( int node_idxi = 0; node_idxi < num_nodes_per_wedge; node_idxi++ )
                            {
                                for ( int dimj = 0; dimj < 3; ++dimj )
                                {
                                    for ( int node_idxj = 0; node_idxj < num_nodes_per_wedge; node_idxj++ )
                                    {
                                        A_tmp[wedge]( node_idxi * 3 + dimi, node_idxj * 3 + dimj ) = A[wedge](
                                            node_idxi + dimi * num_nodes_per_wedge,
                                            node_idxj + dimj * num_nodes_per_wedge );
                                    }
                                }
                            }
                        }
                        reorder_local_dofs( DoFOrdering::DIMENSIONWISE, DoFOrdering::NODEWISE, src[wedge] );
                    }

                    // assemble rotation matrices for boundary nodes
                    // e.g. if we are at CMB, we need to rotate DoFs 0, 1, 2 of each wedge
                    // at SURFACE, we need to rotate DoFs 3, 4, 5

                    constexpr int layer_hex_offset_x[2][3] = { { 0, 1, 0 }, { 1, 0, 1 } };
                    constexpr int layer_hex_offset_y[2][3] = { { 0, 0, 1 }, { 1, 1, 0 } };

                    for ( int wedge = 0; wedge < 2; ++wedge )
                    {
                        // make rotation matrix unity
                        for ( int i = 0; i < LocalMatrixDim; ++i )
                        {
                            R[wedge]( i, i ) = 1.0;
                        }

                        for ( int boundary_node_idx = 0; boundary_node_idx < 3; boundary_node_idx++ )
                        {
                            // compute normal
                            dense::Vec< double, 3 > normal = grid::shell::coords(
                                local_subdomain_id,
                                x_cell + layer_hex_offset_x[wedge][boundary_node_idx],
                                y_cell + layer_hex_offset_y[wedge][boundary_node_idx],
                                r_cell + ( at_cmb ? 0 : 1 ),
                                grid_,
                                radii_ );
                          

                            // compute rotation matrix for DoFs on current node
                            auto R_i = trafo_mat_cartesian_to_normal_tangential( normal );

                            // insert into wedge-local rotation matrix
                            int offset_in_R = at_cmb ? 0 : 9;
                            for ( int dimi = 0; dimi < 3; ++dimi )
                            {
                                for ( int dimj = 0; dimj < 3; ++dimj )
                                {
                                    R[wedge](
                                        offset_in_R + boundary_node_idx * 3 + dimi,
                                        offset_in_R + boundary_node_idx * 3 + dimj ) = R_i( dimi, dimj );
                                }
                            }
                        }

                        // transform local matrix to rotated/ normal-tangential space: pre/post multiply with rotation matrices
                        // TODO transpose this way?
                        A[wedge] = R[wedge] * A_tmp[wedge] * R[wedge].transposed();
                        // transform source dofs to nt-space
                        auto src_tmp = R[wedge] * src[wedge];
                        for ( int i = 0; i < 18; ++i )
                        {
                            src[wedge]( i ) = src_tmp( i );
                        }

                        // eliminate normal components: Dirichlet on the normal-tangential system
                        int node_start = at_surface ? 3 : 0;
                        int node_end   = at_surface ? 6 : 3;

                        for ( int node_idx = node_start; node_idx < node_end; node_idx++ )
                        {
                            int idx = node_idx * 3;
                            /* Eliminate rows and cols for dofs corresponding to the normal component of a velocity */
                            for ( int k = 0; k < 18; ++k )
                            {
                                if ( k != idx )
                                {
                                    boundary_mask( idx, k ) = 0.0;
                                    boundary_mask( k, idx ) = 0.0;
                                }
                            }
                        }
                    }
                }
                else if ( bcf == NEUMANN ) {}
            }

            // apply boundary mask
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                A[wedge].hadamard_product( boundary_mask );
            }

            if ( diagonal_ )
            {
                A[0] = A[0].diagonal();
                A[1] = A[1].diagonal();
            }

            dense::Vec< ScalarT, LocalMatrixDim > dst[num_wedges_per_hex_cell];
            dst[0] = A[0] * src[0];
            dst[1] = A[1] * src[1];

            if ( freeslip_reorder )
            {
                // transform dst back from nt space
                dense::Vec< ScalarT, LocalMatrixDim > dst_tmp[num_wedges_per_hex_cell];
                dst_tmp[0] = R[0].transposed() * dst[0];
                dst_tmp[1] = R[1].transposed() * dst[1];
                for ( int i = 0; i < 18; ++i )
                {
                    dst[0]( i ) = dst_tmp[0]( i );
                    dst[1]( i ) = dst_tmp[1]( i );
                }

                // reorder to dimensionwise ordering
                reorder_local_dofs( DoFOrdering::NODEWISE, DoFOrdering::DIMENSIONWISE, dst[0] );
                reorder_local_dofs( DoFOrdering::NODEWISE, DoFOrdering::DIMENSIONWISE, dst[1] );
            }

            for ( int dimi = 0; dimi < 3; dimi++ )
            {
                dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
                dst_d[0] = dst[0].template slice< 6 >( dimi * num_nodes_per_wedge );
                dst_d[1] = dst[1].template slice< 6 >( dimi * num_nodes_per_wedge );

                atomically_add_local_wedge_vector_coefficients(
                    dst_, local_subdomain_id, x_cell, y_cell, r_cell, dimi, dst_d );
            }
        }
        else
        {
            // ----- FAST PATH (DCA) -----
            //   - Load ALL source dofs (and k) for the team's r-slab into TEAM SHARED MEMORY once.
            //   - Each thread then reads the dofs for its corresponding r_cell (team_rank) from the shared arrays.
            //
            // Team covers radial levels: r_base ... r_base + team_size
            // (need team_size+1 levels because each thread needs r and r+1).

            static constexpr int WEDGE_NODE_OFF[2][6][3] = {
                { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
                { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

            // Map wedge-local node (w, node) -> unique node id in [0..7] that matches your final scatter pattern
            static constexpr int WEDGE_TO_UNIQUE[2][6] = {
                { 0, 1, 2, 3, 4, 5 }, // wedge 0
                { 6, 2, 1, 7, 5, 4 }  // wedge 1
            };

            // ---- single quadrature point collapsed: qp0=qp1=1/3, qp2=0, qw=1 ----
            constexpr double ONE_THIRD      = 1.0 / 3.0;
            constexpr double ONE_SIXTH      = 1.0 / 6.0;
            constexpr double NEG_TWO_THIRDS = -0.66666666666666663;

            // Reference gradients at qp0=qp1=1/3, qp2=0 (constexpr)
            static constexpr double dN_ref[6][3] = {
                { -0.5, -0.5, -ONE_SIXTH },
                { 0.5, 0.0, -ONE_SIXTH },
                { 0.0, 0.5, -ONE_SIXTH },
                { -0.5, -0.5, ONE_SIXTH },
                { 0.5, 0.0, ONE_SIXTH },
                { 0.0, 0.5, ONE_SIXTH } };

            // ---------------------------------------------------------
            // TEAM SCRATCH LAYOUT (UPDATED FOR BANK CONFLICTS):
            //   wedge_surf_phy_coords: [2][3][3]
            //   src_sh:                [4][3][team_size+1]   (xy, dim, level)   <-- level LAST (contiguous across threads)
            //   k_sh:                  [4][team_size+1]      (xy, level)        <-- level LAST
            //
            double* shmem =
                reinterpret_cast< double* >( team.team_shmem().get_shmem( team_shmem_size( team.team_size() ) ) );

            using ScratchCoords = Kokkos::
                View< double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
            using ScratchSrc = Kokkos::
                View< double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
            using ScratchK = Kokkos::
                View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;

            // wedge coords
            ScratchCoords wedge_surf_phy_coords( shmem, 2, 3, 3 );
            shmem += 2 * 3 * 3;

            const int ts   = team.team_size();
            const int nlev = ts + 1;

            // src_sh(xy, dim, level)
            ScratchSrc src_sh( shmem, 4, 3, nlev );
            shmem += 4 * 3 * nlev;

            // k_sh(xy, level)
            ScratchK k_sh( shmem, 4, nlev );
            shmem += 4 * nlev;

            auto r_sh = Kokkos::
                View< double*, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >(
                    shmem, nlev );
            shmem += nlev;

            // ---------------------------------------------------------
            // (1) Load surface xy geometry (once per team) as before
            // ---------------------------------------------------------
            Kokkos::single( Kokkos::PerTeam( team ), [&]() {
                const double q00x = grid_( local_subdomain_id, x_cell, y_cell, 0 );
                const double q00y = grid_( local_subdomain_id, x_cell, y_cell, 1 );
                const double q00z = grid_( local_subdomain_id, x_cell, y_cell, 2 );

                const double q01x = grid_( local_subdomain_id, x_cell, y_cell + 1, 0 );
                const double q01y = grid_( local_subdomain_id, x_cell, y_cell + 1, 1 );
                const double q01z = grid_( local_subdomain_id, x_cell, y_cell + 1, 2 );

                const double q10x = grid_( local_subdomain_id, x_cell + 1, y_cell, 0 );
                const double q10y = grid_( local_subdomain_id, x_cell + 1, y_cell, 1 );
                const double q10z = grid_( local_subdomain_id, x_cell + 1, y_cell, 2 );

                const double q11x = grid_( local_subdomain_id, x_cell + 1, y_cell + 1, 0 );
                const double q11y = grid_( local_subdomain_id, x_cell + 1, y_cell + 1, 1 );
                const double q11z = grid_( local_subdomain_id, x_cell + 1, y_cell + 1, 2 );

                // wedge 0: (q00,q10,q01)
                wedge_surf_phy_coords( 0, 0, 0 ) = q00x;
                wedge_surf_phy_coords( 0, 0, 1 ) = q00y;
                wedge_surf_phy_coords( 0, 0, 2 ) = q00z;
                wedge_surf_phy_coords( 0, 1, 0 ) = q10x;
                wedge_surf_phy_coords( 0, 1, 1 ) = q10y;
                wedge_surf_phy_coords( 0, 1, 2 ) = q10z;
                wedge_surf_phy_coords( 0, 2, 0 ) = q01x;
                wedge_surf_phy_coords( 0, 2, 1 ) = q01y;
                wedge_surf_phy_coords( 0, 2, 2 ) = q01z;

                // wedge 1: (q11,q01,q10)
                wedge_surf_phy_coords( 1, 0, 0 ) = q11x;
                wedge_surf_phy_coords( 1, 0, 1 ) = q11y;
                wedge_surf_phy_coords( 1, 0, 2 ) = q11z;
                wedge_surf_phy_coords( 1, 1, 0 ) = q01x;
                wedge_surf_phy_coords( 1, 1, 1 ) = q01y;
                wedge_surf_phy_coords( 1, 1, 2 ) = q01z;
                wedge_surf_phy_coords( 1, 2, 0 ) = q10x;
                wedge_surf_phy_coords( 1, 2, 1 ) = q10y;
                wedge_surf_phy_coords( 1, 2, 2 ) = q10z;
            } );

            // ---------------------------------------------------------
            // (2) Load ALL k/src dofs for the team's radial slab into shared memory
            //
            // Layout:
            //   level = 0..team_size  (absolute r = r_base + level)
            //   xy    = 0..3 mapped by (dx + 2*dy)
            //
            // Each thread loads its own level = team_rank
            // and the last thread additionally loads level = team_size.
            // ---------------------------------------------------------
            const int r_base = r_block_index * ts;

            auto load_level = [&]( const int level ) {
                const int r_abs = r_base + level;
                r_sh( level )   = radii_( local_subdomain_id, r_abs );

                for ( int dy = 0; dy <= 1; ++dy )
                {
                    for ( int dx = 0; dx <= 1; ++dx )
                    {
                        const int xy = dx + 2 * dy;

                        const int xi = x_cell + dx;
                        const int yi = y_cell + dy;

                        k_sh( xy, level )      = k_( local_subdomain_id, xi, yi, r_abs );
                        src_sh( xy, 0, level ) = src_( local_subdomain_id, xi, yi, r_abs, 0 );
                        src_sh( xy, 1, level ) = src_( local_subdomain_id, xi, yi, r_abs, 1 );
                        src_sh( xy, 2, level ) = src_( local_subdomain_id, xi, yi, r_abs, 2 );
                    }
                }
            };

            // each thread loads its own level
            load_level( team.team_rank() );

            // one extra level (team_size) needed for r+1 of last thread
            if ( team.team_rank() == ts - 1 )
            {
                load_level( ts );
            }

            team.team_barrier();

            // ---------------------------------------------------------
            // Thread-private radii (depends on r_cell)
            // ---------------------------------------------------------
            const int    lvl0 = team.team_rank(); // corresponds to r_cell
            const double r_0  = r_sh( lvl0 );
            const double r_1  = r_sh( lvl0 + 1 );

            // Boundary treatment flags (guard the BC query)
            const bool at_boundary    = at_cmb || at_surface;
            bool       treat_boundary = false;
            if ( at_boundary )
            {
                const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
                treat_boundary              = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
            }

            const int cmb_shift     = ( ( at_boundary && treat_boundary && ( !diagonal_ ) && at_cmb ) ? 3 : 0 );
            const int surface_shift = ( ( at_boundary && treat_boundary && ( !diagonal_ ) && at_surface ) ? 3 : 0 );

            // (1a) unique-node accumulation: 8 nodes per dim
            double dst8[3][8] = { 0.0 };

            // Local level index for this thread

            for ( int w = 0; w < 2; ++w )
            {
                // -------------------------
                // (A) k_eval collapsed from shared memory
                // -------------------------
                double k_sum = 0.0;
#pragma unroll
                for ( int node = 0; node < 6; ++node )
                {
                    const int dx = WEDGE_NODE_OFF[w][node][0];
                    const int dy = WEDGE_NODE_OFF[w][node][1];
                    const int dr = WEDGE_NODE_OFF[w][node][2];

                    const int xy  = dx + 2 * dy;
                    const int lvl = lvl0 + dr;

                    k_sum += k_sh( xy, lvl );
                }
                const double k_eval = ONE_SIXTH * k_sum;

                // -------------------------
                // (B) Jacobian + inv(J)^T
                // -------------------------
                double wJ = 0.0;

                double i00, i01, i02;
                double i10, i11, i12;
                double i20, i21, i22;

                {
                    const double half_dr = 0.5 * ( r_1 - r_0 );
                    const double r_mid   = 0.5 * ( r_0 + r_1 );

                    const double J_0_0 =
                        r_mid * ( -wedge_surf_phy_coords( w, 0, 0 ) + wedge_surf_phy_coords( w, 1, 0 ) );
                    const double J_0_1 =
                        r_mid * ( -wedge_surf_phy_coords( w, 0, 0 ) + wedge_surf_phy_coords( w, 2, 0 ) );
                    const double J_0_2 =
                        half_dr * ( ONE_THIRD * ( wedge_surf_phy_coords( w, 0, 0 ) + wedge_surf_phy_coords( w, 1, 0 ) +
                                                  wedge_surf_phy_coords( w, 2, 0 ) ) );

                    const double J_1_0 =
                        r_mid * ( -wedge_surf_phy_coords( w, 0, 1 ) + wedge_surf_phy_coords( w, 1, 1 ) );
                    const double J_1_1 =
                        r_mid * ( -wedge_surf_phy_coords( w, 0, 1 ) + wedge_surf_phy_coords( w, 2, 1 ) );
                    const double J_1_2 =
                        half_dr * ( ONE_THIRD * ( wedge_surf_phy_coords( w, 0, 1 ) + wedge_surf_phy_coords( w, 1, 1 ) +
                                                  wedge_surf_phy_coords( w, 2, 1 ) ) );

                    const double J_2_0 =
                        r_mid * ( -wedge_surf_phy_coords( w, 0, 2 ) + wedge_surf_phy_coords( w, 1, 2 ) );
                    const double J_2_1 =
                        r_mid * ( -wedge_surf_phy_coords( w, 0, 2 ) + wedge_surf_phy_coords( w, 2, 2 ) );
                    const double J_2_2 =
                        half_dr * ( ONE_THIRD * ( wedge_surf_phy_coords( w, 0, 2 ) + wedge_surf_phy_coords( w, 1, 2 ) +
                                                  wedge_surf_phy_coords( w, 2, 2 ) ) );

                    const double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                                         J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;

                    const double invJ = 1.0 / J_det;

                    // inv(J)^T
                    i00 = invJ * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
                    i01 = invJ * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
                    i02 = invJ * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );

                    i10 = invJ * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
                    i11 = invJ * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
                    i12 = invJ * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );

                    i20 = invJ * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
                    i21 = invJ * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
                    i22 = invJ * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );

                    wJ = Kokkos::abs( J_det ); // qw=1
                }

                const double kwJ = k_eval * wJ;

                // -------------------------
                // (C) grad_u + div_u as scalars (src from shared memory)
                // -------------------------
                {
                    double gu00 = 0.0;
                    double gu10 = 0.0, gu11 = 0.0;
                    double gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
                    double div_u = 0.0;

                    if ( !diagonal_ )
                    {
                        // Assemble gu** and div_u
                        for ( int dimj = 0; dimj < 3; ++dimj )
                        {
#pragma unroll
                            for ( int node_idx = cmb_shift; node_idx < 6 - surface_shift; ++node_idx )
                            {
                                const double gx = dN_ref[node_idx][0];
                                const double gy = dN_ref[node_idx][1];
                                const double gz = dN_ref[node_idx][2];

                                const double g0 = i00 * gx + i01 * gy + i02 * gz;
                                const double g1 = i10 * gx + i11 * gy + i12 * gz;
                                const double g2 = i20 * gx + i21 * gy + i22 * gz;

                                double E00, E11, E22, sym01, sym02, sym12, gdd;
                                column_grad_to_sym( dimj, g0, g1, g2, E00, E11, E22, sym01, sym02, sym12, gdd );

                                const int dx = WEDGE_NODE_OFF[w][node_idx][0];
                                const int dy = WEDGE_NODE_OFF[w][node_idx][1];
                                const int dr = WEDGE_NODE_OFF[w][node_idx][2];

                                const int xy  = dx + 2 * dy;
                                const int lvl = lvl0 + dr;

                                // UPDATED INDEXING: src_sh(xy, dim, level)
                                const double s = src_sh( xy, dimj, lvl );

                                gu00 += E00 * s;
                                gu10 += sym01 * s;
                                gu11 += E11 * s;
                                gu20 += sym02 * s;
                                gu21 += sym12 * s;
                                gu22 += E22 * s;

                                div_u += gdd * s;
                            }
                        }

                        // Pairing -> accumulate into unique-node array dst8
                        for ( int dimi = 0; dimi < 3; ++dimi )
                        {
#pragma unroll
                            for ( int node_idx = cmb_shift; node_idx < 6 - surface_shift; ++node_idx )
                            {
                                const double gx = dN_ref[node_idx][0];
                                const double gy = dN_ref[node_idx][1];
                                const double gz = dN_ref[node_idx][2];

                                const double g0 = i00 * gx + i01 * gy + i02 * gz;
                                const double g1 = i10 * gx + i11 * gy + i12 * gz;
                                const double g2 = i20 * gx + i21 * gy + i22 * gz;

                                double E00, E11, E22, sym01, sym02, sym12, gdd;
                                column_grad_to_sym( dimi, g0, g1, g2, E00, E11, E22, sym01, sym02, sym12, gdd );

                                const double pairing0 = 2.0 * sym01;
                                const double pairing1 = 2.0 * sym02;
                                const double pairing2 = 2.0 * sym12;

                                const int u = WEDGE_TO_UNIQUE[w][node_idx];

                                dst8[dimi][u] += kwJ * ( NEG_TWO_THIRDS * div_u * gdd + 2.0 * pairing0 * gu10 +
                                                         2.0 * pairing1 * gu20 + 2.0 * pairing2 * gu21 +
                                                         2.0 * E00 * gu00 + 2.0 * E11 * gu11 + 2.0 * E22 * gu22 );
                            }
                        }
                    }
                }

                // Diagonal / BC loop -> also accumulate into dst8
                if ( diagonal_ || ( treat_boundary && at_boundary ) )
                {
                    for ( int dim_diagBC = 0; dim_diagBC < 3; ++dim_diagBC )
                    {
#pragma unroll
                        for ( int node_idx = surface_shift; node_idx < 6 - cmb_shift; ++node_idx )
                        {
                            const double gx = dN_ref[node_idx][0];
                            const double gy = dN_ref[node_idx][1];
                            const double gz = dN_ref[node_idx][2];

                            const double g0 = i00 * gx + i01 * gy + i02 * gz;
                            const double g1 = i10 * gx + i11 * gy + i12 * gz;
                            const double g2 = i20 * gx + i21 * gy + i22 * gz;

                            double E00, E11, E22, sym01, sym02, sym12, gdd;
                            column_grad_to_sym( dim_diagBC, g0, g1, g2, E00, E11, E22, sym01, sym02, sym12, gdd );

                            const int dx = WEDGE_NODE_OFF[w][node_idx][0];
                            const int dy = WEDGE_NODE_OFF[w][node_idx][1];
                            const int dr = WEDGE_NODE_OFF[w][node_idx][2];

                            const int xy  = dx + 2 * dy;
                            const int lvl = lvl0 + dr;

                            // UPDATED INDEXING: src_sh(xy, dim, level)
                            const double s = src_sh( xy, dim_diagBC, lvl );

                            const double pairing0 = 4.0 * s;
                            const double pairing1 = 2.0 * s;

                            const int u = WEDGE_TO_UNIQUE[w][node_idx];

                            dst8[dim_diagBC][u] += kwJ * ( pairing0 * ( sym01 * sym01 ) + pairing0 * ( sym02 * sym02 ) +
                                                           pairing0 * ( sym12 * sym12 ) + pairing1 * ( E00 * E00 ) +
                                                           pairing1 * ( E11 * E11 ) + pairing1 * ( E22 * E22 ) +
                                                           NEG_TWO_THIRDS * ( gdd * gdd ) * s );
                        }
                    }
                }
            } // w

            // Final scatter: 8 unique nodes per dim (same result as original merged scatter)

            for ( int dim_add = 0; dim_add < 3; ++dim_add )
            {
                Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell, dim_add ), dst8[dim_add][0] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell, dim_add ), dst8[dim_add][1] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell, dim_add ), dst8[dim_add][2] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1, dim_add ), dst8[dim_add][3] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim_add ), dst8[dim_add][4] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim_add ), dst8[dim_add][5] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, dim_add ), dst8[dim_add][6] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, dim_add ), dst8[dim_add][7] );
            }
        }
    }

    /// @brief: For both trial and test space this function sets up a vector:
    /// each vector element holds the symmetric gradient (a 3x3 matrix) of the shape function of the corresponding dof
    /// (if dimi == dimj, these are the same and we are on the diagonal of the vectorial diffusion operator)
    /// Additionally, we compute the scalar factor for the numerical integral comp: determinant of the jacobian,
    /// evaluation of the coefficient k on the element and the quadrature weight of the current quad-point.

    /// The idea of this function is that the two vectors can be:
    /// - accumulated to the result of the local matvec with 2 * num_nodes_per_wedge complexity
    ///   by scaling the dot product of the trial vec and local src dofs with each element of the test vec
    ///   (and adding to the dst dofs, this is the fused local matvec).
    /// - propagated to the local matrix by an outer product of the two vectors
    ///   (without applying it to dofs). This is e.g. required to assemble the finest grid local
    ///   matrix on-the-fly during GCA/Galerkin coarsening.

    ///
    KOKKOS_INLINE_FUNCTION void assemble_trial_test_vecs(
        const int                               wedge,
        const dense::Vec< ScalarType, VecDim >& quad_point,
        const ScalarType                        quad_weight,
        const ScalarT                           r_1,
        const ScalarT                           r_2,
        dense::Vec< ScalarT, 3 > ( *wedge_phy_surf )[3],
        const dense::Vec< ScalarT, 6 >*           k_local_hex,
        const int                                 dimi,
        const int                                 dimj,
        dense::Mat< ScalarType, VecDim, VecDim >* sym_grad_i,
        dense::Mat< ScalarType, VecDim, VecDim >* sym_grad_j,
        ScalarType&                               jdet_keval_quadweight ) const
    {
        dense::Mat< ScalarType, VecDim, VecDim >       J       = jac( wedge_phy_surf[wedge], r_1, r_2, quad_point );
        const auto                                     det     = J.det();
        const auto                                     abs_det = Kokkos::abs( det );
        const dense::Mat< ScalarType, VecDim, VecDim > J_inv_transposed = J.inv_transposed( det );

        // dot of coeff dofs and element-local shape functions to evaluate the coefficent on the current element
        ScalarType k_eval = 0.0;
        for ( int k = 0; k < num_nodes_per_wedge; k++ )
        {
            k_eval += shape( k, quad_point ) * k_local_hex[wedge]( k );
        }

        for ( int k = 0; k < num_nodes_per_wedge; k++ )
        {
            sym_grad_i[k] = symmetric_grad( J_inv_transposed, quad_point, k, dimi );
            sym_grad_j[k] = symmetric_grad( J_inv_transposed, quad_point, k, dimj );
        }
        jdet_keval_quadweight = quad_weight * k_eval * abs_det;
    }

    /// @brief assemble the local matrix and return it for a given element, wedge, and vectorial component
    /// (determined by dimi, dimj)
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > assemble_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        // Gather surface points for each wedge.
        // TODO gather this for only 1 wedge
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        dense::Vec< ScalarT, 6 > k_local_hex[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( k_local_hex, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

        // Compute the local element matrix.
        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A = {};
        for ( int dimi = 0; dimi < 3; ++dimi )
        {
            for ( int dimj = 0; dimj < 3; ++dimj )
            {
                // spatial dimensions: quadrature points and wedge
                for ( int q = 0; q < num_quad_points; q++ )
                {
                    dense::Mat< ScalarType, VecDim, VecDim > sym_grad_i[num_nodes_per_wedge];
                    dense::Mat< ScalarType, VecDim, VecDim > sym_grad_j[num_nodes_per_wedge];
                    ScalarType                               jdet_keval_quadweight = 0;
                    assemble_trial_test_vecs(
                        wedge,
                        quad_points[q],
                        quad_weights[q],
                        r_1,
                        r_2,
                        wedge_phy_surf,
                        k_local_hex,
                        dimi,
                        dimj,
                        sym_grad_i,
                        sym_grad_j,
                        jdet_keval_quadweight );

                    // propagate on local matrix by outer product of test and trial vecs
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            A( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) +=
                                jdet_keval_quadweight *
                                ( 2 * sym_grad_j[j].double_contract( sym_grad_i[i] ) -
                                  2.0 / 3.0 * sym_grad_j[j]( dimj, dimj ) * sym_grad_i[i]( dimi, dimi ) );
                            // for the div, we just extract the component from the gradient vector
                        }
                    }
                }
            }
        }

      

        return A;
    }

};

static_assert( linalg::GCACapable< EpsilonDivDivKerngen< float > > );
static_assert( linalg::GCACapable< EpsilonDivDivKerngen< double > > );

} // namespace terra::fe::wedge::operators::shell
