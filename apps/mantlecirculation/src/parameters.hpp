#pragma once

#include <string>
#include <variant>

#include "util/cli11_helper.hpp"
#include "util/info.hpp"
#include "util/result.hpp"

namespace terra::mantlecirculation {

struct MeshParameters
{
    int refinement_level_mesh_min   = 1;
    int refinement_level_mesh_max   = 4;
    int refinement_level_subdomains = 0;

    // Nondimensional radii
    double radius_min = 0.5;
    double radius_max = 1.0;
    // Dimensional radii in meter
    double radius_surface_m   = 6371000.0;
    double radius_cmb_m       = 3480000.0;
    double mantle_thickness_m = 2891000.0;
};

struct PlateParameters
{
    bool apply_plate_velocities = false; // This does nothing yet
    int  initial_plate_age      = 400;
    int  final_plate_age        = 0;

    double plate_velocity_scaling = 1.0;
};
struct BoundaryConditionsParameters
{
    enum class VelocityBC
    {
        NO_SLIP,
        FREE_SLIP,
    };

    VelocityBC velocity_bc_cmb     = VelocityBC::NO_SLIP;
    VelocityBC velocity_bc_surface = VelocityBC::NO_SLIP;

    // Nondimensional temperatures
    double temperature_min = 0.0;
    double temperature_max = 1.0;
    // Dimensional temperatures in Kelvin
    double temperature_cmb_K     = 3800.0;
    double temperature_surface_K = 300.0;
    double delta_T_K             = temperature_cmb_K - temperature_surface_K;

    PlateParameters plate_params{};
};

struct ViscosityParameters
{
    bool        radial_profile_enabled       = false;
    std::string radial_profile_csv_filename  = "radial_viscosity_profile.csv";
    std::string radial_profile_radii_key     = "radii";
    std::string radial_profile_viscosity_key = "viscosity";
    double      reference_viscosity          = 1e22;
    double      viscosity                    = 1.0;
};

struct PhysicsParameters
{
    double gravity = 9.81;

    // Non-dimensional numbers
    double rayleigh_number    = 1e5;
    double peclet_number      = 1.0;
    double dissipation_number = 1.0;
    double h_number           = 1.0;

    double thermal_diffusivity     = 1.0;
    double characteristic_velocity = 1e-10; // characteristic diffusive velocity

    double reference_density      = 4500;
    double thermal_expansivity    = 2.5e-5;
    double thermal_conductivity   = 3.0;
    double specific_heat_capacity = 1230;

    bool   internal_heating      = false;
    double internal_heating_rate = 1.0;

    double calc_cm_per_year = 3e-4; // from non-dim velocity to cm/a
    double calc_time_Ma     = 1e6;  // from non-dim time to Ma

    ViscosityParameters viscosity_params{};
};

struct StokesSolverParameters
{
    int    krylov_restart            = 10;
    int    krylov_max_iterations     = 10;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;

    int viscous_pc_num_vcycles                 = 1;
    int viscous_pc_chebyshev_order             = 2;
    int viscous_pc_num_smoothing_steps_prepost = 2;
    int viscous_pc_num_power_iterations        = 10;
};

struct EnergySolverParameters
{
    int    krylov_restart            = 5;
    int    krylov_max_iterations     = 100;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;
};

struct TimeSteppingParameters
{
    double dt_scaling = 0.5;
    double t_end_Ma   = 100.0;
    double t_end      = 1.0;

    int max_timesteps = 10;

    int energy_substeps = 1;
};

struct IOParameters
{
    std::string outdir          = "output";
    bool        overwrite       = false;
    bool        output_pressure = true;

    std::string xdmf_dir                = "xdmf";
    std::string radial_profiles_out_dir = "radial_profiles";
    std::string timer_trees_dir         = "timer_trees";

    std::string checkpoint_dir;
    int         checkpoint_step = -1;
};

// This struct holds options that might be useful for debugging, benchmarking, etc., but are not intended for 'standard' use.
struct DeveloperOptions
{
    bool set_nondimensional_numbers = false;
    bool output_dimensional         = true;
};

struct Parameters
{
    MeshParameters               mesh_params;
    BoundaryConditionsParameters boundary_params;
    StokesSolverParameters       stokes_solver_params;
    EnergySolverParameters       energy_solver_params;
    PhysicsParameters            physics_params;
    TimeSteppingParameters       time_stepping_params;
    IOParameters                 io_params;
    DeveloperOptions             devel_params;

    std::string output_config_file;
};

struct CLIHelp
{};

inline void nondimensionalise( Parameters& prm )
{
    auto& phys     = prm.physics_params;
    auto& mesh     = prm.mesh_params;
    auto& boundary = prm.boundary_params;
    auto& devel    = prm.devel_params;
    auto& time     = prm.time_stepping_params;

    // --- Domain ---

    // radius_max is unchanged from default, always 1.0 per construction
    mesh.radius_min         = mesh.radius_cmb_m / mesh.radius_surface_m;
    mesh.mantle_thickness_m = mesh.radius_surface_m - mesh.radius_cmb_m;

    // --- Boundary conditions ---

    boundary.temperature_min = boundary.temperature_surface_K / boundary.delta_T_K;
    boundary.temperature_max = boundary.temperature_cmb_K / boundary.delta_T_K;

    // Compute characteristic velocity and thermal diffusivity
    phys.characteristic_velocity =
        phys.thermal_conductivity / ( phys.reference_density * phys.specific_heat_capacity * mesh.mantle_thickness_m );

    phys.thermal_diffusivity = phys.thermal_conductivity / ( phys.reference_density * phys.specific_heat_capacity );

    // Precompute conversion factors from non-dim to dimensional quantities
    phys.calc_cm_per_year = phys.characteristic_velocity * 60 * 60 * 24 * 365 * 100; // Velocity in cm/a

    phys.calc_time_Ma = mesh.mantle_thickness_m / ( phys.calc_cm_per_year * 1e4 ); // Time in Ma
    // Acount for plate velocity scaling
    if ( boundary.plate_params.apply_plate_velocities )
    {
        phys.calc_time_Ma /= boundary.plate_params.plate_velocity_scaling;
    }

    // Nondimensionalise time
    time.t_end = time.t_end_Ma / phys.calc_time_Ma;

    if ( !devel.set_nondimensional_numbers )
    {
        // Compute nondimensional numbers
        // Rayleigh number = ( rho * alpha * g * L^3 * dT ) / ( eta * kappa )
        phys.rayleigh_number = ( phys.reference_density * phys.gravity * phys.thermal_expansivity *
                                 std::pow( mesh.mantle_thickness_m, 3 ) * boundary.delta_T_K ) /
                               ( phys.viscosity_params.reference_viscosity * phys.thermal_diffusivity );

        // Peclet number = ( U * L ) / kappa -> should be 1
        phys.peclet_number = ( phys.characteristic_velocity * mesh.mantle_thickness_m ) / phys.thermal_diffusivity;

        // Dissipation number = ( alpha * g * L ) / Cp
        phys.dissipation_number =
            ( phys.thermal_expansivity * phys.gravity * mesh.mantle_thickness_m ) / phys.specific_heat_capacity;

        // H-number = ( H * L ) / ( Cp * U * dT )
        phys.h_number = ( phys.internal_heating_rate * mesh.mantle_thickness_m ) /
                        ( phys.specific_heat_capacity * phys.characteristic_velocity * boundary.delta_T_K );
    }
}

inline util::Result< std::variant< CLIHelp, Parameters > > parse_parameters( int argc, char** argv )
{
    CLI::App app{ "Mantle circulation simulation." };

    Parameters parameters{};

    using util::add_flag_with_default;
    using util::add_option_with_default;

    // Allow config files
    app.set_config( "--config" );

    ///////////////
    /// General ///
    ///////////////

    add_option_with_default(
        app,
        "--write-config-and-exit",
        parameters.output_config_file,
        "Writes a config file with the passed (or default arguments) to the desired location to be then modified and passed. E.g., '--write-config-and-exit my-config.toml'.\n"
        "IMPORTANT: THIS OPTION MUST BE REMOVED IN THE GENERATED CONFIG OR ELSE YOU WILL OVERWRITE IT AGAIN" )
        ->group( "General" );

    ///////////////////////
    /// Domain and mesh ///
    ///////////////////////

    add_option_with_default( app, "--refinement-level-mesh-min", parameters.mesh_params.refinement_level_mesh_min )
        ->group( "Domain" );
    add_option_with_default( app, "--refinement-level-mesh-max", parameters.mesh_params.refinement_level_mesh_max )
        ->group( "Domain" );

    add_option_with_default( app, "--refinement-level-subdomains", parameters.mesh_params.refinement_level_subdomains )
        ->group( "Domain" );

    add_option_with_default( app, "--radius-cmb", parameters.mesh_params.radius_cmb_m )->group( "Domain" );
    add_option_with_default( app, "--radius-surface", parameters.mesh_params.radius_surface_m )->group( "Domain" );

    ///////////////////////////
    /// Boundary conditions ///
    ///////////////////////////

    std::map< std::string, BoundaryConditionsParameters::VelocityBC > velocity_bc_cmb_map{
        { "noslip", BoundaryConditionsParameters::VelocityBC::NO_SLIP },
        { "freeslip", BoundaryConditionsParameters::VelocityBC::FREE_SLIP },
    };

    std::map< std::string, BoundaryConditionsParameters::VelocityBC > velocity_bc_surface_map{
        { "noslip", BoundaryConditionsParameters::VelocityBC::NO_SLIP },
        { "freeslip", BoundaryConditionsParameters::VelocityBC::FREE_SLIP },
    };

    add_option_with_default( app, "--velocity-bc-cmb", parameters.boundary_params.velocity_bc_cmb )
        ->transform( CLI::CheckedTransformer( velocity_bc_cmb_map, CLI::ignore_case ) )
        ->default_val( "noslip" )
        ->group( "Boundary Conditions" );

    add_option_with_default( app, "--velocity-bc-surface", parameters.boundary_params.velocity_bc_surface )
        ->transform( CLI::CheckedTransformer( velocity_bc_surface_map, CLI::ignore_case ) )
        ->default_val( "noslip" )
        ->group( "Boundary Conditions" );

    add_option_with_default( app, "--temperature-cmb", parameters.boundary_params.temperature_cmb_K )
        ->group( "Boundary Conditions" );

    add_option_with_default( app, "--temperature-surface", parameters.boundary_params.temperature_surface_K )
        ->group( "Boundary Conditions" );

    //////////////////////////////
    /// (Geo-)Physical parameters ///
    //////////////////////////////
    add_flag_with_default( app, "--internal-heating-enabled", parameters.physics_params.internal_heating );
    add_option_with_default( app, "--internal-heating-rate", parameters.physics_params.internal_heating_rate );

    add_option_with_default( app, "--reference-density", parameters.physics_params.reference_density );
    add_option_with_default( app, "--thermal-expansivity", parameters.physics_params.thermal_expansivity );
    add_option_with_default( app, "--thermal-conductivity", parameters.physics_params.thermal_conductivity );
    add_option_with_default( app, "--specific-heat-capacity", parameters.physics_params.specific_heat_capacity );

    // Viscosity parameters
    add_option_with_default(
        app, "--reference-viscosity", parameters.physics_params.viscosity_params.reference_viscosity )
        ->group( "Viscosity" );
    const auto radial_profile_enabled =
        add_flag_with_default(
            app, "--viscosity-radial-profile", parameters.physics_params.viscosity_params.radial_profile_enabled )
            ->group( "Viscosity" )
            ->description(
                "Add this flag if you want to supply a radial viscosity profile. "
                "Then use further flags/arguments (starting with --viscosity-radial-profile-<...>) to specify the file path etc. "
                "If you omit this flag, the viscosity is set to const (eta = 1)." );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-csv-filename",
        parameters.physics_params.viscosity_params.radial_profile_csv_filename )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-radii-key",
        parameters.physics_params.viscosity_params.radial_profile_radii_key )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-value-key",
        parameters.physics_params.viscosity_params.radial_profile_viscosity_key )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );

    ///////////////////////////
    /// Time discretization ///
    ///////////////////////////

    add_option_with_default( app, "--dt-scaling", parameters.time_stepping_params.dt_scaling )
        ->description(
            "A robust (stable) dt is computed the the actual face-normal velocity fluxes and cell volumes via a "
            "parallel reduce over all cells. However, a smaller value might still be desired due to accuracy "
            "considerations. You can scale the computed dt using this value (e.g. set to 0.5 to half the estimated dt, "
            "set to 1.0 to just use the estimated dt)." )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--t-end", parameters.time_stepping_params.t_end_Ma )
        ->group( "Time Discretization" )
        ->description( "Final time in Ma." );
    add_option_with_default( app, "--max-timesteps", parameters.time_stepping_params.max_timesteps )
        ->group( "Time Discretization" )
        ->description(
            "Simulation aborts when this time step index is reached. "
            "If a checkpoint is loaded, the simulation will start at the next step after the loaded checkpoint. "
            "This means the number of time steps executed might be smaller than what is passed in here." );
    add_option_with_default( app, "--energy-substeps", parameters.time_stepping_params.energy_substeps )
        ->group( "Time Discretization" );

    /////////////////////
    /// Stokes solver ///
    /////////////////////

    add_option_with_default( app, "--stokes-krylov-restart", parameters.stokes_solver_params.krylov_restart )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-max-iterations", parameters.stokes_solver_params.krylov_max_iterations )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-relative-tolerance", parameters.stokes_solver_params.krylov_relative_tolerance )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-absolute-tolerance", parameters.stokes_solver_params.krylov_absolute_tolerance )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-viscous-pc-num-vcycles", parameters.stokes_solver_params.viscous_pc_num_vcycles )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-viscous-pc-cheby-order", parameters.stokes_solver_params.viscous_pc_chebyshev_order )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app,
        "--stokes-viscous-pc-num-smoothing-steps-prepost",
        parameters.stokes_solver_params.viscous_pc_num_smoothing_steps_prepost )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app,
        "--stokes-viscous-pc-num-power-iterations",
        parameters.stokes_solver_params.viscous_pc_num_power_iterations )
        ->group( "Stokes Solver" );

    /////////////////////
    /// Energy solver ///
    /////////////////////

    add_option_with_default( app, "--energy-krylov-restart", parameters.energy_solver_params.krylov_restart )
        ->group( "Energy Solver" );
    add_option_with_default(
        app, "--energy-krylov-max-iterations", parameters.energy_solver_params.krylov_max_iterations )
        ->group( "Energy Solver" );
    add_option_with_default(
        app, "--energy-krylov-relative-tolerance", parameters.energy_solver_params.krylov_relative_tolerance )
        ->group( "Energy Solver" );
    add_option_with_default(
        app, "--energy-krylov-absolute-tolerance", parameters.energy_solver_params.krylov_absolute_tolerance )
        ->group( "Energy Solver" );

    //////////////////////
    /// Input / output ///
    //////////////////////

    add_option_with_default( app, "--outdir", parameters.io_params.outdir )->group( "I/O" );
    add_flag_with_default( app, "--outdir-overwrite", parameters.io_params.overwrite )->group( "I/O" );
    add_option_with_default( app, "--output-pressure", parameters.io_params.output_pressure )->group( "I/O" );

    add_option_with_default( app, "--checkpoint-dir", parameters.io_params.checkpoint_dir )->group( "I/O" );
    add_option_with_default( app, "--checkpoint-step", parameters.io_params.checkpoint_step )->group( "I/O" );

    try
    {
        app.parse( argc, argv );
    }
    catch ( const CLI::ParseError& e )
    {
        app.exit( e );
        if ( e.get_exit_code() == static_cast< int >( CLI::ExitCodes::Success ) )
        {
            return { CLIHelp{} };
        }
        return { "CLI parse error" };
    }

    // Nondimensionalise all relevant input parameters
    nondimensionalise( parameters );

    util::logroot << "=========================================\n";
    util::logroot << "     Starting mantle circulation app     \n";
    util::logroot << "     Run with -h or --help for help      \n";
    util::logroot << "=========================================\n";

    util::print_general_info( argc, argv, util::logroot );
    util::print_cli_summary( app, util::logroot );
    util::logroot << std::endl;

    if ( !parameters.output_config_file.empty() )
    {
        util::logroot << "Writing config file to " << parameters.output_config_file << " and exiting." << std::endl;
        std::ofstream config_file( parameters.output_config_file );
        config_file << app.config_to_str( true, true );
    }

    return { parameters };
}

}; // namespace terra::mantlecirculation
