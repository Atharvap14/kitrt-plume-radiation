// Benchmark RadLib property models against shipped LBL reference curves.
//
// This standalone tool intentionally lives outside KiT-RT's main CMake build:
// compile it against an external RadLib installation when running validation.

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad.h"
#include "rad_planck_mean.h"
#include "rad_rcslw.h"
#include "rad_wsgg.h"

using std::string;
using std::vector;

namespace {

struct CaseConfig {
    string name;
    double pressure_pa = 101325.0;
    double hot_length_m = 0.5;
    int ntheta = 101;
    int nx_hot = 1001;
    vector<double> cold_lengths_m{0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0};
};

struct FieldState {
    vector<double> temperature_k;
    vector<double> soot_volume_fraction;
    vector<double> x_h2o;
    vector<double> x_co2;
    vector<double> x_co;
    vector<double> x_ch4;
    double rcslw_reference_temperature_k = 1000.0;
    double rcslw_reference_x_h2o = 0.0;
    double rcslw_reference_x_co2 = 0.0;
    double rcslw_reference_x_co = 0.0;
};

void intensity_implicit_trapezoid(const vector<double> &x,
                                  double theta,
                                  const vector<double> &temperature_k,
                                  const vector<vector<double>> &kabs,
                                  const vector<vector<double>> &awts,
                                  const vector<double> &source_weight,
                                  const vector<double> &ilo,
                                  const vector<double> &ihi,
                                  vector<vector<double>> &intensity) {
    const int nx = static_cast<int>(x.size());
    const int nbands = static_cast<int>(intensity[0].size());
    const double mu = std::abs(std::cos(theta));

    if (theta <= M_PI / 2.0) {
        intensity[0] = ilo;
        for (int i = 0; i < nx - 1; ++i) {
            const double ib_next = rad::sigma / M_PI * std::pow(temperature_k[i + 1], 4.0);
            const double ib_here = rad::sigma / M_PI * std::pow(temperature_k[i], 4.0);
            const double dx = x[i + 1] - x[i];
            for (int j = 0; j < nbands; ++j) {
                intensity[i + 1][j] =
                    (intensity[i][j] +
                     dx / mu * 0.5 *
                         (kabs[i + 1][j] * source_weight[i + 1] * awts[i + 1][j] * ib_next +
                          kabs[i][j] * (source_weight[i] * awts[i][j] * ib_here - intensity[i][j]))) /
                    (1.0 + dx / mu * 0.5 * kabs[i + 1][j]);
            }
        }
        return;
    }

    intensity[nx - 1] = ihi;
    for (int i = nx - 1; i > 0; --i) {
        const double ib_prev = rad::sigma / M_PI * std::pow(temperature_k[i - 1], 4.0);
        const double ib_here = rad::sigma / M_PI * std::pow(temperature_k[i], 4.0);
        const double dx = x[i] - x[i - 1];
        for (int j = 0; j < nbands; ++j) {
            intensity[i - 1][j] =
                (intensity[i][j] +
                 dx / mu * 0.5 *
                     (kabs[i - 1][j] * source_weight[i - 1] * awts[i - 1][j] * ib_prev +
                      kabs[i][j] * (source_weight[i] * awts[i][j] * ib_here - intensity[i][j]))) /
                (1.0 + dx / mu * 0.5 * kabs[i - 1][j]);
        }
    }
}

vector<double> solve_parallel_planes_with_properties(const vector<vector<double>> &kabs,
                                                     const vector<vector<double>> &awts,
                                                     const vector<double> &source_weight,
                                                     double length_m,
                                                     int ntheta,
                                                     const FieldState &field) {
    const int nx = static_cast<int>(field.temperature_k.size());
    const double dx = length_m / static_cast<double>(nx - 1);
    vector<double> x(nx, 0.0);
    for (int i = 1; i < nx; ++i) {
        x[i] = x[i - 1] + dx;
    }

    const int nbands = static_cast<int>(kabs[0].size());
    const vector<double> ilo(nbands, 0.0);
    const vector<double> ihi(nbands, 0.0);
    vector<vector<double>> intensity(nx, vector<double>(nbands, 0.0));
    vector<double> flux(nx, 0.0);
    const double dtheta = M_PI / static_cast<double>(ntheta);

    for (int j = 0; j < ntheta; ++j) {
        const double theta = dtheta * (static_cast<double>(j) + 0.5);
        intensity_implicit_trapezoid(x, theta, field.temperature_k, kabs, awts, source_weight, ilo, ihi, intensity);
        for (int i = 0; i < nx; ++i) {
            double band_sum = 0.0;
            for (int k = 0; k < nbands; ++k) {
                band_sum += intensity[i][k];
            }
            flux[i] += 2.0 * M_PI * dtheta * std::cos(theta) * std::sin(theta) * band_sum;
        }
    }
    return flux;
}

void fill_properties(rad &property_model,
                     const FieldState &field,
                     double pressure_pa,
                     vector<vector<double>> &kabs,
                     vector<vector<double>> &awts) {
    const int nx = static_cast<int>(field.temperature_k.size());
    for (int i = 0; i < nx; ++i) {
        property_model.get_k_a(kabs[i],
                               awts[i],
                               field.temperature_k[i],
                               pressure_pa,
                               field.soot_volume_fraction[i],
                               field.x_h2o[i],
                               field.x_co2[i],
                               field.x_co[i],
                               field.x_ch4[i]);
    }
}

vector<double> solve_parallel_planes(rad &property_model,
                                     double length_m,
                                     int ntheta,
                                     const FieldState &field,
                                     double pressure_pa) {
    const int nx = static_cast<int>(field.temperature_k.size());
    const int nbands = property_model.get_nGGa();
    vector<vector<double>> kabs(nx, vector<double>(nbands, 0.0));
    vector<vector<double>> awts(nx, vector<double>(nbands, 0.0));
    vector<double> source_weight(nx, 1.0);
    fill_properties(property_model, field, pressure_pa, kabs, awts);
    return solve_parallel_planes_with_properties(kabs, awts, source_weight, length_m, ntheta, field);
}

vector<double> solve_parallel_planes_local_rcslw(const CaseConfig &config,
                                                 double length_m,
                                                 int nGG,
                                                 const FieldState &field) {
    const int nx = static_cast<int>(field.temperature_k.size());
    const int nbands = nGG + 1;
    vector<double> total_flux(nx, 0.0);

    auto add_reference = [&](rad_rcslw &model, const vector<double> &source_weight) {
        vector<vector<double>> kabs(nx, vector<double>(nbands, 0.0));
        vector<vector<double>> awts(nx, vector<double>(nbands, 0.0));
        fill_properties(model, field, config.pressure_pa, kabs, awts);
        vector<double> partial_flux =
            solve_parallel_planes_with_properties(kabs, awts, source_weight, length_m, config.ntheta, field);
        for (int i = 0; i < nx; ++i) {
            total_flux[i] += partial_flux[i];
        }
    };

    if (config.name == "S1") {
        rad_rcslw hot_model(nGG, 2000.0, config.pressure_pa, 0.0, 0.2, 0.1, 0.0);
        rad_rcslw cold_model(nGG, 300.0, config.pressure_pa, 0.0, 0.2, 0.1, 0.0);
        vector<double> hot_source(nx, 0.0);
        vector<double> cold_source(nx, 0.0);
        for (int i = 0; i < nx; ++i) {
            if (field.temperature_k[i] > 1000.0) {
                hot_source[i] = 1.0;
            } else {
                cold_source[i] = 1.0;
            }
        }
        add_reference(hot_model, hot_source);
        add_reference(cold_model, cold_source);
    } else if (config.name == "S2") {
        rad_rcslw rich_model(nGG, 1000.0, config.pressure_pa, 0.0, 0.0, 0.4, 0.0);
        rad_rcslw lean_model(nGG, 1000.0, config.pressure_pa, 0.0, 0.0, 0.1, 0.0);
        vector<double> rich_source(nx, 0.0);
        vector<double> lean_source(nx, 0.0);
        for (int i = 0; i < nx; ++i) {
            if (field.x_co2[i] > 0.25) {
                rich_source[i] = 1.0;
            } else {
                lean_source[i] = 1.0;
            }
        }
        add_reference(rich_model, rich_source);
        add_reference(lean_model, lean_source);
    } else {
        throw std::runtime_error("Local RCSLW is only implemented for S1/S2");
    }
    return total_flux;
}

FieldState build_case_field(const CaseConfig &config, double cold_length_m) {
    const double length_m = config.hot_length_m + cold_length_m;
    const int nx = static_cast<int>(config.nx_hot * length_m / config.hot_length_m);
    const double dx = length_m / static_cast<double>(nx - 1);

    FieldState field;
    field.temperature_k.assign(nx, 1000.0);
    field.soot_volume_fraction.assign(nx, 0.0);
    field.x_h2o.assign(nx, 0.0);
    field.x_co2.assign(nx, 0.0);
    field.x_co.assign(nx, 0.0);
    field.x_ch4.assign(nx, 0.0);

    if (config.name == "S1") {
        const double hot_temperature_k = 2000.0;
        const double cold_temperature_k = 300.0;
        field.rcslw_reference_temperature_k =
            (hot_temperature_k * config.hot_length_m + cold_temperature_k * cold_length_m) / length_m;
        field.rcslw_reference_x_h2o = 0.2;
        field.rcslw_reference_x_co2 = 0.1;

        double x = 0.0;
        for (int i = 0; i < nx; ++i) {
            field.temperature_k[i] = (x <= config.hot_length_m) ? hot_temperature_k : cold_temperature_k;
            field.x_h2o[i] = 0.2;
            field.x_co2[i] = 0.1;
            x += dx;
        }
        return field;
    }

    if (config.name == "S2") {
        const double xco2_hot_zone = 0.4;
        const double xco2_cold_zone = 0.1;
        field.rcslw_reference_temperature_k = 1000.0;
        field.rcslw_reference_x_co2 =
            (xco2_hot_zone * config.hot_length_m + xco2_cold_zone * cold_length_m) / length_m;

        double x = 0.0;
        for (int i = 0; i < nx; ++i) {
            field.temperature_k[i] = 1000.0;
            field.x_co2[i] = (x <= config.hot_length_m) ? xco2_hot_zone : xco2_cold_zone;
            x += dx;
        }
        return field;
    }

    throw std::runtime_error("Unsupported case: " + config.name);
}

std::unique_ptr<rad> make_property_model(const string &method, const FieldState &field, int nGG, double pressure_pa) {
    if (method == "planckmean") {
        return std::make_unique<rad_planck_mean>();
    }
    if (method == "wsgg") {
        return std::make_unique<rad_wsgg>();
    }
    if (method == "rcslw") {
        return std::make_unique<rad_rcslw>(nGG,
                                           field.rcslw_reference_temperature_k,
                                           pressure_pa,
                                           0.0,
                                           field.rcslw_reference_x_h2o,
                                           field.rcslw_reference_x_co2,
                                           field.rcslw_reference_x_co);
    }
    if (method == "rcslw_local") {
        return nullptr;
    }
    throw std::runtime_error("Unsupported method: " + method);
}

double normalize_flux(const string &case_name, double flux) {
    if (case_name == "S1") {
        return flux / rad::sigma / std::pow(2000.0, 4.0);
    }
    return flux / rad::sigma / std::pow(1000.0, 4.0);
}

string read_arg(int argc, char **argv, const string &name, const string &fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

} // namespace

int main(int argc, char **argv) {
    try {
        CaseConfig config;
        config.name = read_arg(argc, argv, "--case", "S2");
        const string method = read_arg(argc, argv, "--method", "rcslw");
        const int nGG = std::stoi(read_arg(argc, argv, "--ngg", "24"));

        const auto t0 = std::chrono::steady_clock::now();
        vector<double> normalized_fluxes;
        normalized_fluxes.reserve(config.cold_lengths_m.size());

        for (double cold_length_m : config.cold_lengths_m) {
            const double length_m = config.hot_length_m + cold_length_m;
            FieldState field = build_case_field(config, cold_length_m);
            vector<double> flux;
            if (method == "rcslw_local") {
                flux = solve_parallel_planes_local_rcslw(config, length_m, nGG, field);
            } else {
                std::unique_ptr<rad> property_model = make_property_model(method, field, nGG, config.pressure_pa);
                flux = solve_parallel_planes(*property_model, length_m, config.ntheta, field, config.pressure_pa);
            }
            normalized_fluxes.push_back(normalize_flux(config.name, flux.back()));
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "meta,case,method,nGG,elapsed_s\n";
        std::cout << "meta," << config.name << "," << method << "," << nGG << "," << elapsed_s << "\n";
        std::cout << "data,Lcold_m,normalized_flux\n";
        for (std::size_t i = 0; i < config.cold_lengths_m.size(); ++i) {
            std::cout << "data," << config.cold_lengths_m[i] << "," << normalized_fluxes[i] << "\n";
        }
    } catch (const std::exception &exc) {
        std::cerr << "radlib_flux_benchmark error: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
