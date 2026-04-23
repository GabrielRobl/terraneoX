
#pragma once

#include <fstream>
#include <vector>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "io/xdmf.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "parameters.hpp"
#include "shell/radial_profiles.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/result.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using util::logroot;
using util::Ok;
using util::Result;

using terra::kernels::common::scale;

using ScalarType = double;

namespace terra::mantlecirculation {

inline Result<> create_directories( const IOParameters& io_parameters )
{
    const auto xdmf_dir            = io_parameters.outdir + "/" + io_parameters.xdmf_dir;
    const auto radial_profiles_dir = io_parameters.outdir + "/" + io_parameters.radial_profiles_out_dir;
    const auto timer_trees_dir     = io_parameters.outdir + "/" + io_parameters.timer_trees_dir;

    if ( !io_parameters.overwrite && std::filesystem::exists( io_parameters.outdir ) )
    {
        return { "Will not overwrite existing directory (to not accidentally delete old simulation data). "
                 "Use -h for help and look for an overwrite option or choose a different output dir name." };
    }

    util::prepare_empty_directory( io_parameters.outdir );
    util::prepare_empty_directory( xdmf_dir );
    util::prepare_empty_directory( radial_profiles_dir );
    util::prepare_empty_directory( timer_trees_dir );

    return { Ok{} };
}

inline Result<> write_xdmf(
    std::optional< io::XDMFOutput< ScalarType > >& xdmf_output,
    std::optional< io::XDMFOutput< ScalarType > >& xdmf_output_pressure,
    const Parameters&                              prm,
    Grid4DDataScalar< ScalarType >&                Temperature_data,
    Grid4DDataVec< ScalarType, 3 >&                Velocity_data,
    Grid4DDataScalar< ScalarType >&                Viscosity_data,
    Grid4DDataScalar< ScalarType >&                Pressure_data )
{
    if ( prm.devel_params.output_dimensional )
    {
        // Redimensionalise ...
        scale( Temperature_data, prm.boundary_params.delta_T_K );
        scale( Velocity_data, prm.physics_params.calc_cm_per_year );
        scale( Viscosity_data, prm.physics_params.viscosity_params.reference_viscosity );

        xdmf_output->write();

        // ... and nondimensionalise again.
        scale( Temperature_data, 1.0 / prm.boundary_params.delta_T_K );
        scale( Velocity_data, 1.0 / prm.physics_params.calc_cm_per_year );
        scale( Viscosity_data, 1.0 / prm.physics_params.viscosity_params.reference_viscosity );

        // Redim, write and nondim pressure
        if ( xdmf_output_pressure )
        {
            scale(
                Pressure_data,
                ( prm.physics_params.viscosity_params.reference_viscosity *
                  prm.physics_params.characteristic_velocity ) /
                    prm.mesh_params.mantle_thickness_m );

            xdmf_output_pressure->write();

            scale(
                Pressure_data,
                prm.mesh_params.mantle_thickness_m / ( prm.physics_params.viscosity_params.reference_viscosity *
                                                       prm.physics_params.characteristic_velocity ) );
        }
    }
    else
    {
        xdmf_output->write();
        if ( xdmf_output_pressure )
            xdmf_output_pressure->write();
    }

    return { Ok{} };
}

inline Result<> compute_and_write_radial_profiles(
    const VectorQ1Scalar< ScalarType >& scalar_function,
    const Grid2DDataScalar< int >&      subdomain_shell_idx,
    const DistributedDomain&            domain,
    const IOParameters&                 io_parameters,
    const int                           timestep )
{
    const auto profiles = shell::radial_profiles_to_table< ScalarType >(
        shell::radial_profiles(
            scalar_function, subdomain_shell_idx, static_cast< int >( domain.domain_info().radii().size() ) ),
        domain.domain_info().radii() );

    if ( mpi::rank() == 0 )
    {
        std::ofstream out(
            io_parameters.outdir + "/" + io_parameters.radial_profiles_out_dir + "/radial_profiles_" +
            scalar_function.grid_data().label() + "_" + std::to_string( timestep ) + ".csv" );
        profiles.print_csv( out );
    }

    return { Ok{} };
}

inline Result<> write_timer_tree( const IOParameters& io_parameters, const int timestep )
{
    util::TimerTree::instance().aggregate_mpi();
    if ( mpi::rank() == 0 )
    {
        const auto timer_tree_file = io_parameters.outdir + "/" + io_parameters.timer_trees_dir + "/timer_tree_" +
                                     std::to_string( timestep ) + ".json";
        logroot << "Writing timer tree to " << timer_tree_file << std::endl;
        std::ofstream out( timer_tree_file );
        out << util::TimerTree::instance().json_aggregate();
        out.close();
    }

    return { Ok{} };
}

} // namespace terra::mantlecirculation
