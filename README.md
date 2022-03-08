To compile the Cython functions:

python3 setup_quad_sum.py build_ext --inplace \
python3 setup_deriv.py build_ext --inplace

To run Primodal:

python3 new_main.py > output.dat

To visualise and test the output, use
https://github.com/pc559/primodal_helper_functions/blob/main/source/working_with_coeffs_example.ipynb

The default model is DBI, edit config.py to change to a canonical kinetic term (labelled by "malda").
In config.py kinks and resonance oscillations can be added to the potential, or new scenarios
can be defined.
