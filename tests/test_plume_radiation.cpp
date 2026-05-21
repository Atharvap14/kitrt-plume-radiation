#include "catch.hpp"

#include "common/config.hpp"
#include "common/globalconstants.hpp"
#include "solvers/plumedtmsolver.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

TEST_CASE( "Plume homogeneous ray exact solution", "[plume]" ) {
    const double length      = 2.0;
    const double kappa       = 0.25;
    const double temperature = 1000.0;
    const double source      = PlumeDTMSolver::BlackbodyIntensity( temperature );

    const double intensity = PlumeDTMSolver::IntegrateHomogeneousRay( length, kappa, temperature );
    REQUIRE( intensity == Approx( source * ( 1.0 - std::exp( -kappa * length ) ) ).epsilon( 1e-12 ) );

    REQUIRE( PlumeDTMSolver::IntegrateHomogeneousRay( length, 0.0, temperature ) == Approx( 0.0 ) );
    REQUIRE( PlumeDTMSolver::IntegrateHomogeneousRay( length, 0.0, temperature, 42.0 ) == Approx( 42.0 ) );
}

TEST_CASE( "Plume spectral gas-cell fixture reproduces grouped slab flux", "[plume]" ) {
    const std::vector<PlumeDTMSolver::SpectralGroup> groups = {
        { 20.0, 3.0 },
        { 0.5, 0.75 },
        { 0.0, 10.0 },
    };
    const double length = 0.01;
    const double expectedFlux =
        static_cast<double>( PI ) * ( 3.0 * ( 1.0 - std::exp( -20.0 * length ) ) + 0.75 * ( 1.0 - std::exp( -0.5 * length ) ) );

    REQUIRE( PlumeDTMSolver::IntegrateSpectralGasCellFlux( groups, length ) == Approx( expectedFlux ).epsilon( 1e-12 ) );
}

TEST_CASE( "Plume spectral gas-cell CSV round trip", "[plume]" ) {
    const std::string filename = std::string( TESTS_PATH ) + "result/plume_spectral_gas_cell.csv";
    std::filesystem::create_directories( std::filesystem::path( filename ).parent_path() );

    std::ofstream out( filename );
    out << "case,group,kappa_1_per_m,source_radiance_W_m2_sr,path_length_m,gt_flux_W_m2\n";
    out << "synthetic,0,2.0,4.0,0.5,7.0\n";
    out << "synthetic,1,1.0,3.0,0.5,7.0\n";
    out.close();

    const auto fixture = PlumeDTMSolver::LoadSpectralGasCellCSV( filename );
    REQUIRE( fixture.groups.size() == 2 );
    REQUIRE( fixture.pathLengthM == Approx( 0.5 ) );
    REQUIRE( fixture.gtFluxWm2 == Approx( 7.0 ) );

    const double expectedFlux = static_cast<double>( PI ) * ( 4.0 * ( 1.0 - std::exp( -1.0 ) ) + 3.0 * ( 1.0 - std::exp( -0.5 ) ) );
    REQUIRE( PlumeDTMSolver::IntegrateSpectralGasCellFlux( fixture.groups, fixture.pathLengthM ) == Approx( expectedFlux ).epsilon( 1e-12 ) );
}

TEST_CASE( "Plume synthetic field CSV round trip", "[plume]" ) {
    const std::string filename = std::string( TESTS_PATH ) + "result/plume_synthetic_field.csv";
    std::filesystem::create_directories( std::filesystem::path( filename ).parent_path() );

    PlumeDTMSolver::WriteSyntheticFieldCSV( filename, 0.0, 1.0, 2.0, 3, 4, 875.0, 0.3 );
    const auto field = PlumeDTMSolver::LoadFieldCSV( filename );

    REQUIRE( field.zValues.size() == 3 );
    REQUIRE( field.rValues.size() == 4 );

    double temperature = 0.0;
    double kappa       = 0.0;
    REQUIRE( field.Interpolate( 0.5, 1.0, temperature, kappa ) );
    REQUIRE( temperature == Approx( 875.0 ) );
    REQUIRE( kappa == Approx( 0.3 ) );
}

TEST_CASE( "Plume CPU DTM synthetic fixture converges to constant-intensity flux", "[plume]" ) {
    const std::string configFile = std::string( TESTS_PATH ) + "input/unit_tests/plume/plume_synthetic.cfg";

    auto config = std::make_unique<Config>( configFile );
    PlumeDTMSolver solver( config.get() );
    solver.Solve();

    const auto& samples = solver.GetHeatFluxSamples();
    REQUIRE( samples.size() == 1 );

    const double intensity = PlumeDTMSolver::IntegrateHomogeneousRay( config->GetPlumeRayMaxDistance(),
                                                                      config->GetPlumeSyntheticKappa(),
                                                                      config->GetPlumeSyntheticTemperature() );
    const double exactFlux = static_cast<double>( PI ) * intensity;

    REQUIRE( samples[0].radius == Approx( 0.0 ) );
    REQUIRE( samples[0].heatFlux == Approx( exactFlux ).epsilon( 1e-3 ) );

    const std::string outputCsv = config->GetOutputFile() + "_plume_heat_flux.csv";
    std::ifstream out( outputCsv );
    REQUIRE( out.good() );
}
