#include "solvers/plumespectralgascell.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
struct Args {
    std::string fixture;
    std::string output;
    double lengthM = std::numeric_limits<double>::quiet_NaN();
    double gtFlux  = std::numeric_limits<double>::quiet_NaN();
};

void PrintUsage( const char* executable ) {
    std::cerr << "Usage: " << executable << " --fixture kitrt_groups.csv [--length-m value] [--gt-flux value] [--out output.csv]\n";
}

Args ParseArgs( int argc, char** argv ) {
    Args args;
    for( int idx = 1; idx < argc; ++idx ) {
        const std::string key = argv[idx];
        auto needValue        = [&]( const std::string& name ) -> std::string {
            if( idx + 1 >= argc ) throw std::runtime_error( "Missing value for " + name );
            return argv[++idx];
        };

        if( key == "--fixture" ) {
            args.fixture = needValue( key );
        }
        else if( key == "--length-m" ) {
            args.lengthM = std::stod( needValue( key ) );
        }
        else if( key == "--gt-flux" ) {
            args.gtFlux = std::stod( needValue( key ) );
        }
        else if( key == "--out" ) {
            args.output = needValue( key );
        }
        else if( key == "--help" || key == "-h" ) {
            PrintUsage( argv[0] );
            std::exit( 0 );
        }
        else {
            throw std::runtime_error( "Unknown argument: " + key );
        }
    }
    if( args.fixture.empty() ) throw std::runtime_error( "--fixture is required" );
    return args;
}
}    // namespace

int main( int argc, char** argv ) {
    try {
        const Args args = ParseArgs( argc, argv );
        const auto fixture = PlumeSpectralGasCell::LoadFixtureCSV( args.fixture );
        const double lengthM = std::isnan( args.lengthM ) ? fixture.pathLengthM : args.lengthM;
        const double gtFlux  = std::isnan( args.gtFlux ) ? fixture.gtFluxWm2 : args.gtFlux;

        if( std::isnan( lengthM ) || lengthM <= 0.0 ) {
            throw std::runtime_error( "No positive path length was provided by --length-m or fixture path_length_m." );
        }

        const double flux = PlumeSpectralGasCell::IntegrateHomogeneousSlabFlux( fixture.groups, lengthM );
        const double relError = std::isnan( gtFlux ) ? std::numeric_limits<double>::quiet_NaN() : std::abs( flux - gtFlux ) / std::max( std::abs( gtFlux ), 1e-300 );

        std::ostream* out = &std::cout;
        std::ofstream file;
        if( !args.output.empty() ) {
            const std::filesystem::path outputPath( args.output );
            if( outputPath.has_parent_path() ) std::filesystem::create_directories( outputPath.parent_path() );
            file.open( args.output );
            if( !file ) throw std::runtime_error( "Unable to open output CSV: " + args.output );
            out = &file;
        }

        *out << std::setprecision( 17 );
        *out << "model,groups,path_length_m,hemispherical_flux_W_m2,gt_flux_W_m2,flux_rel_error\n";
        *out << "KiT-RT spectral gas-cell," << fixture.groups.size() << "," << lengthM << "," << flux << ",";
        if( std::isnan( gtFlux ) ) {
            *out << ",";
        }
        else {
            *out << gtFlux << ",";
        }
        if( std::isnan( relError ) ) {
            *out << "\n";
        }
        else {
            *out << relError << "\n";
        }
    }
    catch( const std::exception& exc ) {
        std::cerr << "plume_spectral_gas_cell: " << exc.what() << "\n";
        PrintUsage( argv[0] );
        return 1;
    }
    return 0;
}
