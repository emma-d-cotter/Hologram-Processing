# holographic camera parameters
pix_size = 7400e-9 # edge size of a pixel (in m)
wavelength = 658e-9 # laser wavelength ( in m)
# holographic camera geometry
L_air = 0.230 # laser distance from camera to water (in m)
L_water = 1 # separation between housings (in m)

gl_sigma = 1.2 # parameter for std_corr focus metric

# indices of refraction
n_air = 1.0003 # index of refraction in air
# calculate index of refraction in water (from Bashkatov 2003)
wavelength_nm = wavelength*1e9
A = 1.32074
B = 5207.924
C = -2.55522e8
D = 9.35006
n_water = A + B/wavelength_nm**2 + C/wavelength_nm**4 + D/wavelength_nm**6

