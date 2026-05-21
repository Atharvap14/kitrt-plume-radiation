Plume Non-Gray Validation Notes
================================

Goal
----

The first plume ray sweep is gray. The next physics step is to replace the
single ``kappa_1_per_m`` field with spectral groups:

.. code-block:: text

   I = sum_j I_j
   dI_j/ds = -kappa_j I_j + kappa_j a_j(T, composition) I_b(T)

The current Python sweep has a ``--spectrum demo-nongray`` option that exercises
this multi-group transport path with synthetic weights and absorption scales.
It is useful for MLX/MPI/CPU performance testing, but its coefficients are not
validated plume-gas properties.

Most Useful Validation Sources
------------------------------

1. Fraga, Bordbar, Hostikka, and Franca, 2020

   * Title: "Benchmark Solutions of Three-Dimensional Radiative Transfer in
     Nongray Media Using Line-by-Line Integration"
   * DOI: https://doi.org/10.1115/1.4045666
   * Open manuscript:
     https://acris.aalto.fi/ws/portalfiles/portal/42621552/ENG_Fraga_et_al_Benchmark_solutions_Journal_of_Heat_Transfer_yhdistetty.pdf
   * Why it matters: publishes line-by-line HITEMP2010 benchmark values for
     non-gray H2O, H2O-N2, and CO2-H2O-N2 cases, including wall heat flux and
     volumetric source terms. This is the cleanest near-term table-driven
     validation target.

2. RadLib

   * Code: https://github.com/BYUignite/radlib
   * Documentation: https://ignite.byu.edu/radlib_documentation/
   * Paper: https://doi.org/10.1016/j.cpc.2021.108227
   * Mendeley data/package: https://doi.org/10.17632/rs5kvnr86r.1
   * Why it matters: MIT-licensed C++ radiation-property library with Python
     and Fortran interfaces. It includes Planck mean, WSGG, and rank-correlated
     SLW models, plus 1D ray-tracing examples compared to line-by-line data.
     This is the best candidate for a practical property-model dependency.
   * Local validation command after building RadLib externally:

     .. code-block:: bash

        python tools/python/validate_radlib_flux.py --case ex_S2 --radlib-root /path/to/radlib

     On the Apple M3 test machine, the RadLib ex_S2 RCSLW-24 final-flux curve
     matched the shipped LBL reference with max relative error 2.56e-3 and mean
     relative error 9.22e-4.

     To compare speed/fidelity for Planck mean, WSGG, and RCSLW group counts
     across the S1 and S2 LBL references:

     .. code-block:: bash

        python tools/python/benchmark_radlib_methods.py --radlib-root /path/to/radlib

     In the initial benchmark, S2 converged to max relative flux error around
     2.5e-3 with RCSLW-24/25, while S1 plateaued around 1.6e-1 even with more
     groups. That S1 behavior indicates spectral-model/reference-state bias,
     not Monte Carlo noise or an insufficient ray count.

     A source-split local RCSLW experiment was also added. It transports hot
     and cold/source-region emissions with separate reference states, then sums
     the resulting fluxes. This improved S2 at 16 groups, but worsened S1
     relative to the original single-reference RCSLW. Therefore the plume solver
     should not be wired to this local RCSLW variant as a production model.
     S1 needs a stronger treatment such as true full-spectrum correlated-k or
     line-by-line/RADIS-derived tabulation.

3. RADIS

   * Code: https://github.com/radis/radis
   * Documentation: https://radis.readthedocs.io/
   * Why it matters: open-source line-by-line spectra for HITRAN, HITEMP, and
     ExoMol species. It can generate spectral absorption/emission data for
     H2O, CO2, CO, and other gases, and can serve as a reference generator for
     smaller validation fixtures.

4. HITRAN HAPI

   * Documentation: https://hitran.org/hapi/
   * Code: https://github.com/hitranonline/hapi
   * Why it matters: official Python API for HITRAN line data and absorption,
     transmittance, and radiance spectra. Useful for small gas-cell validation
     cases; API-key access is required for downloads.

5. Axisymmetric jet-flame LBL studies

   * Chu, Consalvi, Gu, and Liu, 2017:
     https://doi.org/10.1016/j.jqsrt.2017.02.008
   * Centeno, Brittes, Coelho, and Franca, 2015:
     https://doi.org/10.1016/j.jqsrt.2015.02.006
   * Centeno, Brittes, Rodrigues, Coelho, and Franca, 2018:
     https://doi.org/10.1016/j.ijheatmasstransfer.2018.02.040
   * Why they matter: closest geometry match to an axisymmetric plume. They are
     good scientific targets, but less immediately plug-and-play than RadLib or
     Fraga et al. because the underlying fields/results may need digitization or
     author-provided data.

Implementation Path
-------------------

* Short term: keep ``demo-nongray`` for backend performance and convergence
  tests.
* Validation v1: add Fraga et al. tabulated wall-flux/source-term fixtures and
  compare a rectangular-domain DTM case against those line-by-line values.
* Property model v1: add an optional RadLib adapter for WSGG/RCSLW groups.
* Reference generator: use RADIS or HAPI to generate small, versioned gas-cell
  spectra for H2O/CO2/CO and compare band-integrated transmission/emission.
* Plume validation: once plume composition fields are available, map
  ``temperature_K``, species mole fractions, and pressure to spectral groups;
  then compare integrated base heat flux against LBL/RadLib/RADIS fixtures
  before doing large MLX/CUDA data generation.
