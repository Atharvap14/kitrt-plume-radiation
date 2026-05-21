#ifndef PLUMESPECTRALGASCELL_H
#define PLUMESPECTRALGASCELL_H

#include "common/globalconstants.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace PlumeSpectralGasCell
{
struct SpectralGroup {
    double kappa1PerM;
    double sourceRadianceWm2Sr;
};

struct Fixture {
    std::vector<SpectralGroup> groups;
    double pathLengthM = std::numeric_limits<double>::quiet_NaN();
    double gtFluxWm2   = std::numeric_limits<double>::quiet_NaN();
};

inline std::string Trim( const std::string& text ) {
    const std::string whitespace = " \t\n\r";
    const auto begin             = text.find_first_not_of( whitespace );
    if( begin == std::string::npos ) return "";
    const auto end = text.find_last_not_of( whitespace );
    return text.substr( begin, end - begin + 1 );
}

inline std::vector<std::string> SplitCSV( const std::string& line ) {
    std::vector<std::string> values;
    std::stringstream stream( line );
    std::string item;
    while( std::getline( stream, item, ',' ) ) {
        values.push_back( Trim( item ) );
    }
    return values;
}

inline double ParseDoubleColumn( const std::vector<std::string>& values,
                                 const std::map<std::string, unsigned>& columns,
                                 const std::string& name ) {
    const auto it = columns.find( name );
    if( it == columns.end() || it->second >= values.size() ) {
        throw std::runtime_error( "Missing spectral gas-cell CSV column: " + name );
    }
    return std::stod( values[it->second] );
}

inline Fixture LoadFixtureCSV( const std::string& filename ) {
    std::ifstream in( filename );
    if( !in ) throw std::runtime_error( "Unable to open spectral gas-cell CSV: " + filename );

    std::string line;
    if( !std::getline( in, line ) ) throw std::runtime_error( "Spectral gas-cell CSV is empty: " + filename );

    const std::vector<std::string> header = SplitCSV( line );
    std::map<std::string, unsigned> columns;
    for( unsigned i = 0; i < header.size(); ++i ) columns[header[i]] = i;

    const std::vector<std::string> required = { "kappa_1_per_m", "source_radiance_W_m2_sr" };
    for( const auto& name : required ) {
        if( columns.find( name ) == columns.end() ) throw std::runtime_error( "Missing spectral gas-cell CSV column: " + name );
    }

    Fixture fixture;
    while( std::getline( in, line ) ) {
        if( Trim( line ).empty() ) continue;
        const auto values = SplitCSV( line );
        if( values.size() < header.size() ) throw std::runtime_error( "Malformed spectral gas-cell CSV row: " + line );

        const double kappa = ParseDoubleColumn( values, columns, "kappa_1_per_m" );
        const double source = ParseDoubleColumn( values, columns, "source_radiance_W_m2_sr" );
        if( kappa < 0.0 ) throw std::runtime_error( "Spectral gas-cell kappa values must be non-negative." );
        if( source < 0.0 ) throw std::runtime_error( "Spectral gas-cell source radiance values must be non-negative." );

        fixture.groups.push_back( { kappa, source } );

        if( columns.find( "path_length_m" ) != columns.end() ) {
            const double value = ParseDoubleColumn( values, columns, "path_length_m" );
            if( std::isnan( fixture.pathLengthM ) ) fixture.pathLengthM = value;
        }
        if( columns.find( "gt_flux_W_m2" ) != columns.end() ) {
            const double value = ParseDoubleColumn( values, columns, "gt_flux_W_m2" );
            if( std::isnan( fixture.gtFluxWm2 ) ) fixture.gtFluxWm2 = value;
        }
    }

    if( fixture.groups.empty() ) throw std::runtime_error( "Spectral gas-cell CSV does not contain data rows: " + filename );
    return fixture;
}

inline double IntegrateHomogeneousSlabRadiance( const std::vector<SpectralGroup>& groups, double lengthM ) {
    if( lengthM <= 0.0 ) return 0.0;

    double radiance = 0.0;
    for( const auto& group : groups ) {
        if( group.kappa1PerM <= 0.0 ) continue;
        const double tau         = group.kappa1PerM * lengthM;
        const double attenuation = std::exp( -std::min( tau, 700.0 ) );
        radiance += group.sourceRadianceWm2Sr * ( 1.0 - attenuation );
    }
    return radiance;
}

inline double IntegrateHomogeneousSlabFlux( const std::vector<SpectralGroup>& groups, double lengthM ) {
    return static_cast<double>( PI ) * IntegrateHomogeneousSlabRadiance( groups, lengthM );
}
}    // namespace PlumeSpectralGasCell

#endif
