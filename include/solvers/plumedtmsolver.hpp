#ifndef PLUMEDTMSOLVER_H
#define PLUMEDTMSOLVER_H

#include "solvers/plumespectralgascell.hpp"

#include <string>
#include <vector>

class Config;

class PlumeDTMSolver
{
  public:
    struct FieldData {
        std::vector<double> zValues;
        std::vector<double> rValues;
        std::vector<double> temperature;
        std::vector<double> kappa;

        bool Interpolate( double z, double r, double& temperatureOut, double& kappaOut ) const;
        double TemperatureAt( unsigned idxZ, unsigned idxR ) const;
        double KappaAt( unsigned idxZ, unsigned idxR ) const;
    };

    struct HeatFluxSample {
        double radius;
        double heatFlux;
    };

    using SpectralGroup        = PlumeSpectralGasCell::SpectralGroup;
    using SpectralGasCellInput = PlumeSpectralGasCell::Fixture;

    explicit PlumeDTMSolver( Config* settings );

    void Solve();
    const std::vector<HeatFluxSample>& GetHeatFluxSamples() const { return _heatFluxSamples; }

    static double BlackbodyIntensity( double temperature );
    static double IntegrateHomogeneousRay( double length, double kappa, double temperature, double incomingIntensity = 0.0 );
    static SpectralGasCellInput LoadSpectralGasCellCSV( const std::string& filename );
    static double IntegrateSpectralGasCellFlux( const std::vector<SpectralGroup>& groups, double lengthM );
    static FieldData LoadFieldCSV( const std::string& filename );
    static FieldData MakeUniformField( double zMin,
                                       double zMax,
                                       double rMax,
                                       unsigned long zCells,
                                       unsigned long rCells,
                                       double temperature,
                                       double kappa );
    static void WriteSyntheticFieldCSV( const std::string& filename,
                                        double zMin,
                                        double zMax,
                                        double rMax,
                                        unsigned long zCells,
                                        unsigned long rCells,
                                        double temperature,
                                        double kappa );

  private:
    Config* _settings;
    FieldData _field;
    std::vector<HeatFluxSample> _heatFluxSamples;

    FieldData LoadOrCreateField() const;
    double ComputeHeatFluxAtRadius( double radius ) const;
    double IntegrateRay( double baseRadius, double theta, double phi ) const;
    bool IsNozzleShadowed( double z, double r ) const;
    void WriteCSV() const;
    void WriteVTK() const;
};

#endif
