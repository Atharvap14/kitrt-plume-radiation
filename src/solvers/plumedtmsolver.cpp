#include "solvers/plumedtmsolver.hpp"

#include "common/config.hpp"
#include "common/globalconstants.hpp"
#include "toolboxes/errormessages.hpp"

#include "spdlog/spdlog.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

namespace
{
std::string Trim( const std::string& text ) {
    const std::string whitespace = " \t\n\r";
    const auto begin             = text.find_first_not_of( whitespace );
    if( begin == std::string::npos ) return "";
    const auto end = text.find_last_not_of( whitespace );
    return text.substr( begin, end - begin + 1 );
}

std::vector<std::string> SplitCSV( const std::string& line ) {
    std::vector<std::string> values;
    std::stringstream stream( line );
    std::string item;
    while( std::getline( stream, item, ',' ) ) {
        values.push_back( Trim( item ) );
    }
    return values;
}

bool FindBracket( const std::vector<double>& values, double x, unsigned& lower, unsigned& upper, double& t ) {
    if( values.empty() || x < values.front() || x > values.back() ) return false;
    if( values.size() == 1u || x <= values.front() ) {
        lower = 0u;
        upper = 0u;
        t     = 0.0;
        return true;
    }
    if( x >= values.back() ) {
        lower = static_cast<unsigned>( values.size() - 1u );
        upper = lower;
        t     = 0.0;
        return true;
    }

    auto it = std::upper_bound( values.begin(), values.end(), x );
    upper   = static_cast<unsigned>( std::distance( values.begin(), it ) );
    lower   = upper - 1u;
    t       = ( x - values[lower] ) / ( values[upper] - values[lower] );
    return true;
}

double Lerp( double a, double b, double t ) { return a + t * ( b - a ); }
}    // namespace

double PlumeDTMSolver::FieldData::TemperatureAt( unsigned idxZ, unsigned idxR ) const {
    return temperature[idxZ * rValues.size() + idxR];
}

double PlumeDTMSolver::FieldData::KappaAt( unsigned idxZ, unsigned idxR ) const { return kappa[idxZ * rValues.size() + idxR]; }

bool PlumeDTMSolver::FieldData::Interpolate( double z, double r, double& temperatureOut, double& kappaOut ) const {
    unsigned z0 = 0u;
    unsigned z1 = 0u;
    unsigned r0 = 0u;
    unsigned r1 = 0u;
    double tz   = 0.0;
    double tr   = 0.0;

    if( !FindBracket( zValues, z, z0, z1, tz ) || !FindBracket( rValues, r, r0, r1, tr ) ) return false;

    const double t00 = TemperatureAt( z0, r0 );
    const double t10 = TemperatureAt( z1, r0 );
    const double t01 = TemperatureAt( z0, r1 );
    const double t11 = TemperatureAt( z1, r1 );
    const double k00 = KappaAt( z0, r0 );
    const double k10 = KappaAt( z1, r0 );
    const double k01 = KappaAt( z0, r1 );
    const double k11 = KappaAt( z1, r1 );

    temperatureOut = Lerp( Lerp( t00, t10, tz ), Lerp( t01, t11, tz ), tr );
    kappaOut       = Lerp( Lerp( k00, k10, tz ), Lerp( k01, k11, tz ), tr );
    return true;
}

PlumeDTMSolver::PlumeDTMSolver( Config* settings ) : _settings( settings ) {}

double PlumeDTMSolver::BlackbodyIntensity( double temperature ) {
    return static_cast<double>( STEFAN_BOLTZMANN_CONSTANT / PI ) * std::pow( temperature, 4.0 );
}

double PlumeDTMSolver::IntegrateHomogeneousRay( double length, double kappa, double temperature, double incomingIntensity ) {
    if( length <= 0.0 ) return incomingIntensity;
    if( kappa <= 0.0 ) return incomingIntensity;

    const double tau         = kappa * length;
    const double attenuation = std::exp( -std::min( tau, 700.0 ) );
    const double source      = BlackbodyIntensity( temperature );
    return incomingIntensity * attenuation + source * ( 1.0 - attenuation );
}

PlumeDTMSolver::SpectralGasCellInput PlumeDTMSolver::LoadSpectralGasCellCSV( const std::string& filename ) {
    return PlumeSpectralGasCell::LoadFixtureCSV( filename );
}

double PlumeDTMSolver::IntegrateSpectralGasCellFlux( const std::vector<SpectralGroup>& groups, double lengthM ) {
    return PlumeSpectralGasCell::IntegrateHomogeneousSlabFlux( groups, lengthM );
}

PlumeDTMSolver::FieldData PlumeDTMSolver::MakeUniformField( double zMin,
                                                            double zMax,
                                                            double rMax,
                                                            unsigned long zCells,
                                                            unsigned long rCells,
                                                            double temperature,
                                                            double kappa ) {
    if( zCells < 2u || rCells < 2u ) ErrorMessages::Error( "Synthetic plume fields require at least two z and r nodes.", CURRENT_FUNCTION );
    if( zMax <= zMin ) ErrorMessages::Error( "Synthetic plume zMax must be larger than zMin.", CURRENT_FUNCTION );
    if( rMax <= 0.0 ) ErrorMessages::Error( "Synthetic plume rMax must be positive.", CURRENT_FUNCTION );
    if( temperature < 0.0 || kappa < 0.0 ) ErrorMessages::Error( "Synthetic plume temperature and kappa must be non-negative.", CURRENT_FUNCTION );

    FieldData field;
    field.zValues.resize( zCells );
    field.rValues.resize( rCells );
    for( unsigned long i = 0; i < zCells; ++i ) {
        field.zValues[i] = zMin + ( zMax - zMin ) * static_cast<double>( i ) / static_cast<double>( zCells - 1u );
    }
    for( unsigned long i = 0; i < rCells; ++i ) {
        field.rValues[i] = rMax * static_cast<double>( i ) / static_cast<double>( rCells - 1u );
    }

    field.temperature.assign( zCells * rCells, temperature );
    field.kappa.assign( zCells * rCells, kappa );
    return field;
}

void PlumeDTMSolver::WriteSyntheticFieldCSV( const std::string& filename,
                                             double zMin,
                                             double zMax,
                                             double rMax,
                                             unsigned long zCells,
                                             unsigned long rCells,
                                             double temperature,
                                             double kappa ) {
    const FieldData field = MakeUniformField( zMin, zMax, rMax, zCells, rCells, temperature, kappa );

    const std::filesystem::path outPath( filename );
    if( outPath.has_parent_path() ) std::filesystem::create_directories( outPath.parent_path() );

    std::ofstream out( filename );
    if( !out ) ErrorMessages::Error( "Unable to write synthetic plume field CSV: " + filename, CURRENT_FUNCTION );

    out << std::setprecision( 17 );
    out << "z_m,r_m,temperature_K,kappa_1_per_m\n";
    for( unsigned iz = 0; iz < field.zValues.size(); ++iz ) {
        for( unsigned ir = 0; ir < field.rValues.size(); ++ir ) {
            out << field.zValues[iz] << "," << field.rValues[ir] << "," << field.TemperatureAt( iz, ir ) << "," << field.KappaAt( iz, ir )
                << "\n";
        }
    }
}

PlumeDTMSolver::FieldData PlumeDTMSolver::LoadFieldCSV( const std::string& filename ) {
    std::ifstream in( filename );
    if( !in ) ErrorMessages::Error( "Unable to open plume field CSV: " + filename, CURRENT_FUNCTION );

    std::string line;
    if( !std::getline( in, line ) ) ErrorMessages::Error( "Plume field CSV is empty: " + filename, CURRENT_FUNCTION );

    const std::vector<std::string> header = SplitCSV( line );
    std::map<std::string, unsigned> columns;
    for( unsigned i = 0; i < header.size(); ++i ) columns[header[i]] = i;

    const std::vector<std::string> required = { "z_m", "r_m", "temperature_K", "kappa_1_per_m" };
    for( const auto& name : required ) {
        if( columns.find( name ) == columns.end() ) ErrorMessages::Error( "Missing plume field CSV column: " + name, CURRENT_FUNCTION );
    }

    struct Row {
        double z;
        double r;
        double temperature;
        double kappa;
    };
    std::vector<Row> rows;
    std::vector<double> zValues;
    std::vector<double> rValues;

    while( std::getline( in, line ) ) {
        if( Trim( line ).empty() ) continue;
        const auto values = SplitCSV( line );
        if( values.size() < header.size() ) ErrorMessages::Error( "Malformed plume field CSV row: " + line, CURRENT_FUNCTION );

        Row row{};
        row.z           = std::stod( values[columns["z_m"]] );
        row.r           = std::stod( values[columns["r_m"]] );
        row.temperature = std::stod( values[columns["temperature_K"]] );
        row.kappa       = std::stod( values[columns["kappa_1_per_m"]] );
        if( row.r < 0.0 ) ErrorMessages::Error( "Plume field radius values must be non-negative.", CURRENT_FUNCTION );
        if( row.temperature < 0.0 ) ErrorMessages::Error( "Plume field temperature values must be non-negative.", CURRENT_FUNCTION );
        if( row.kappa < 0.0 ) ErrorMessages::Error( "Plume field kappa values must be non-negative.", CURRENT_FUNCTION );

        rows.push_back( row );
        zValues.push_back( row.z );
        rValues.push_back( row.r );
    }

    if( rows.empty() ) ErrorMessages::Error( "Plume field CSV does not contain data rows: " + filename, CURRENT_FUNCTION );

    std::sort( zValues.begin(), zValues.end() );
    zValues.erase( std::unique( zValues.begin(), zValues.end() ), zValues.end() );
    std::sort( rValues.begin(), rValues.end() );
    rValues.erase( std::unique( rValues.begin(), rValues.end() ), rValues.end() );

    FieldData field;
    field.zValues     = zValues;
    field.rValues     = rValues;
    const auto nValues = zValues.size() * rValues.size();
    field.temperature.assign( nValues, std::numeric_limits<double>::quiet_NaN() );
    field.kappa.assign( nValues, std::numeric_limits<double>::quiet_NaN() );

    for( const auto& row : rows ) {
        const auto iz = static_cast<unsigned>( std::distance( zValues.begin(), std::find( zValues.begin(), zValues.end(), row.z ) ) );
        const auto ir = static_cast<unsigned>( std::distance( rValues.begin(), std::find( rValues.begin(), rValues.end(), row.r ) ) );
        const auto id = iz * rValues.size() + ir;
        field.temperature[id] = row.temperature;
        field.kappa[id]       = row.kappa;
    }

    for( unsigned i = 0; i < nValues; ++i ) {
        if( std::isnan( field.temperature[i] ) || std::isnan( field.kappa[i] ) ) {
            ErrorMessages::Error( "Plume field CSV must contain every point of the structured (z,r) grid.", CURRENT_FUNCTION );
        }
    }

    return field;
}

PlumeDTMSolver::FieldData PlumeDTMSolver::LoadOrCreateField() const {
    if( _settings->GetPlumeSyntheticField() ) {
        return MakeUniformField( _settings->GetPlumeBaseZ(),
                                 _settings->GetPlumeBaseZ() + _settings->GetPlumeSyntheticZMax(),
                                 _settings->GetPlumeSyntheticRMax(),
                                 _settings->GetPlumeSyntheticZCells(),
                                 _settings->GetPlumeSyntheticRCells(),
                                 _settings->GetPlumeSyntheticTemperature(),
                                 _settings->GetPlumeSyntheticKappa() );
    }
    return LoadFieldCSV( _settings->GetPlumeFieldFile() );
}

void PlumeDTMSolver::Solve() {
    if( _settings->GetPlumeBackend() == PLUME_BACKEND_CUDA ) {
        ErrorMessages::Error( "PLUME_BACKEND=CUDA is reserved for the next milestone; use PLUME_BACKEND=CPU for v1 validation.", CURRENT_FUNCTION );
    }

    _field = LoadOrCreateField();
    _heatFluxSamples.clear();
    _heatFluxSamples.reserve( _settings->GetPlumeBaseSamples() );

    const unsigned long nBase = _settings->GetPlumeBaseSamples();
    for( unsigned long idx = 0; idx < nBase; ++idx ) {
        const double radius =
            ( nBase == 1u ) ? 0.0 : _settings->GetPlumeBaseRadius() * static_cast<double>( idx ) / static_cast<double>( nBase - 1u );
        _heatFluxSamples.push_back( { radius, ComputeHeatFluxAtRadius( radius ) } );
    }

    WriteCSV();
    if( _settings->GetPlumeWriteVTK() ) WriteVTK();

    if( auto log = spdlog::get( "event" ) ) {
        log->info( "| PlumeRadiation wrote {} base heat-flux samples to {}", _heatFluxSamples.size(), _settings->GetOutputFile() + "_plume_heat_flux.csv" );
    }
}

double PlumeDTMSolver::ComputeHeatFluxAtRadius( double radius ) const {
    if( _settings->GetPlumeNozzleShadow() && radius <= _settings->GetPlumeNozzleRadius() ) return 0.0;

    const unsigned long nTheta = _settings->GetPlumeRayPolarSamples();
    const unsigned long nPhi   = _settings->GetPlumeRayAzimuthSamples();
    const double dTheta        = static_cast<double>( 0.5L * PI ) / static_cast<double>( nTheta );
    const double dPhi          = static_cast<double>( 2.0L * PI ) / static_cast<double>( nPhi );

    double heatFlux = 0.0;
    for( unsigned long idxTheta = 0; idxTheta < nTheta; ++idxTheta ) {
        const double theta    = ( static_cast<double>( idxTheta ) + 0.5 ) * dTheta;
        const double sinTheta = std::sin( theta );
        const double cosTheta = std::cos( theta );
        for( unsigned long idxPhi = 0; idxPhi < nPhi; ++idxPhi ) {
            const double phi       = ( static_cast<double>( idxPhi ) + 0.5 ) * dPhi;
            const double intensity = IntegrateRay( radius, theta, phi );
            heatFlux += intensity * cosTheta * sinTheta * dTheta * dPhi;
        }
    }
    return heatFlux;
}

double PlumeDTMSolver::IntegrateRay( double baseRadius, double theta, double phi ) const {
    const double sinTheta = std::sin( theta );
    const double dirX     = sinTheta * std::cos( phi );
    const double dirY     = sinTheta * std::sin( phi );
    const double dirZ     = std::cos( theta );

    double intensity    = 0.0;
    double transmittance = 1.0;

    for( double s = 0.0; s < _settings->GetPlumeRayMaxDistance(); s += _settings->GetPlumeRayStep() ) {
        const double ds   = std::min( _settings->GetPlumeRayStep(), _settings->GetPlumeRayMaxDistance() - s );
        const double smid = s + 0.5 * ds;
        const double x    = baseRadius + dirX * smid;
        const double y    = dirY * smid;
        const double z    = _settings->GetPlumeBaseZ() + dirZ * smid;
        const double r    = std::sqrt( x * x + y * y );

        if( _settings->GetPlumeNozzleShadow() && IsNozzleShadowed( z, r ) ) break;

        double temperature = 0.0;
        double kappa       = 0.0;
        if( !_field.Interpolate( z, r, temperature, kappa ) ) break;

        const double tau         = kappa * ds;
        const double attenuation = std::exp( -std::min( tau, 700.0 ) );
        intensity += transmittance * BlackbodyIntensity( temperature ) * ( 1.0 - attenuation );
        transmittance *= attenuation;

        if( transmittance < 1e-14 ) break;
    }

    return intensity + transmittance * _settings->GetPlumeIncomingIntensity();
}

bool PlumeDTMSolver::IsNozzleShadowed( double z, double r ) const {
    return z >= _settings->GetPlumeBaseZ() && z <= _settings->GetPlumeBaseZ() + _settings->GetPlumeNozzleLength() &&
           r <= _settings->GetPlumeNozzleRadius();
}

void PlumeDTMSolver::WriteCSV() const {
    const std::string filename = _settings->GetOutputFile() + "_plume_heat_flux.csv";
    std::ofstream out( filename );
    if( !out ) ErrorMessages::Error( "Unable to write plume heat-flux CSV: " + filename, CURRENT_FUNCTION );

    out << std::setprecision( 17 );
    out << "radius_m,heat_flux_W_m2\n";
    for( const auto& sample : _heatFluxSamples ) {
        out << sample.radius << "," << sample.heatFlux << "\n";
    }
}

void PlumeDTMSolver::WriteVTK() const {
    const std::string filename = _settings->GetOutputFile() + "_plume_heat_flux.vtk";
    std::ofstream out( filename );
    if( !out ) ErrorMessages::Error( "Unable to write plume heat-flux VTK: " + filename, CURRENT_FUNCTION );

    out << std::setprecision( 17 );
    out << "# vtk DataFile Version 3.0\n";
    out << "KiT-RT PlumeRadiation base heat flux\n";
    out << "ASCII\n";
    out << "DATASET POLYDATA\n";
    out << "POINTS " << _heatFluxSamples.size() << " double\n";
    for( const auto& sample : _heatFluxSamples ) {
        out << sample.radius << " 0 0\n";
    }
    if( !_heatFluxSamples.empty() ) {
        out << "LINES 1 " << _heatFluxSamples.size() + 1u << "\n";
        out << _heatFluxSamples.size();
        for( unsigned i = 0; i < _heatFluxSamples.size(); ++i ) out << " " << i;
        out << "\n";
    }
    out << "POINT_DATA " << _heatFluxSamples.size() << "\n";
    out << "SCALARS heat_flux_W_m2 double 1\n";
    out << "LOOKUP_TABLE default\n";
    for( const auto& sample : _heatFluxSamples ) {
        out << sample.heatFlux << "\n";
    }
}
