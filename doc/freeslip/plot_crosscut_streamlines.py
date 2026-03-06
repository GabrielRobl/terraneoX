import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt, pi
from matplotlib.patches import FancyArrowPatch

c_Y = sqrt(15.0 / (32.0 * pi))

def Y22(theta, phi):
    return c_Y * np.sin(theta)**2 * np.cos(2*phi)

def dY22_dtheta(theta, phi):
    return c_Y * 2 * np.sin(theta) * np.cos(theta) * np.cos(2*phi)

def make_solution(A, B, C, D, E):
    def Pl(r):
        return A*r**2 + B*r**(-3) + C*r**4 + D*r**(-1) + E*r**5
    def dPldr(r):
        return 2*A*r - 3*B*r**(-4) + 4*C*r**3 - D*r**(-2) + 5*E*r**4
    def u_r(r, theta, phi):
        return -6 * Pl(r) * Y22(theta, phi) / r
    def u_theta(r, theta, phi):
        return -(Pl(r)/r + dPldr(r)) * dY22_dtheta(theta, phi)
    return u_r, u_theta

# ZeroSlip
ur_ns, uth_ns = make_solution(
    5.08347602739726064230e-03, 9.81021689497716840520e-05,
    -1.13441780821917810596e-02, -7.81844558599695542596e-04,
    6.94444444444444405895e-03)

# FreeZeroSlip
ur_fs, uth_fs = make_solution(
    4.54414476717381228543e-03, 3.74596588289534304281e-05,
    -1.10448363301060385744e-02, -4.81212540341170963258e-04,
    6.94444444444444405895e-03)

Rm, Rp = 0.5, 1.0
phi0 = 0.0
vmax_global = 5.5e-4

# =========================================================================
fig = plt.figure(figsize=(20, 24))

# Layout: 3 rows
# Row 0: full cross-sections (3 panels)
# Row 1: zoomed quiver near CMB (2 panels) + radial profile (1 panel)
# Row 2: zoomed quiver near surface (2 panels) + radial profile (1 panel)
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# --- Regular grid for full cross-section ---
nx, nz = 500, 500
xi = np.linspace(-Rp, Rp, nx)
zi = np.linspace(-Rp, Rp, nz)
Xi, Zi = np.meshgrid(xi, zi)
Ri = np.sqrt(Xi**2 + Zi**2)
Ti = np.arccos(np.clip(Zi / np.where(Ri > 1e-14, Ri, 1e-14), -1, 1))
Phi_i = np.where(Xi >= 0, 0.0, np.pi)
mask = (Ri < Rm * 0.99) | (Ri > Rp * 1.01)

cases = [
    (ur_ns, uth_ns, 'ZeroSlip (ns/ns)'),
    (ur_fs, uth_fs, 'FreeZeroSlip (fs/ns)'),
]

# --- Row 0: Full cross-sections with streamlines ---
for idx, (ur_fn, uth_fn, title) in enumerate(cases):
    ax = fig.add_subplot(gs[0, idx])

    ur_i = ur_fn(Ri, Ti, Phi_i)
    uth_i = uth_fn(Ri, Ti, Phi_i)
    sign_x = np.where(Xi >= 0, 1.0, -1.0)
    Ux_i = sign_x * (ur_i * np.sin(Ti) + uth_i * np.cos(Ti))
    Uz_i = ur_i * np.cos(Ti) - uth_i * np.sin(Ti)
    vmag_i = np.sqrt(Ux_i**2 + Uz_i**2)
    Ux_i[mask] = np.nan
    Uz_i[mask] = np.nan
    vmag_i[mask] = np.nan

    im = ax.pcolormesh(Xi, Zi, vmag_i, cmap='inferno', shading='auto',
                       vmin=0, vmax=vmax_global, zorder=0)
    speed_i = np.sqrt(np.nan_to_num(Ux_i)**2 + np.nan_to_num(Uz_i)**2)
    lw = 2.0 * speed_i / (vmax_global + 1e-20)
    ax.streamplot(xi, zi, Ux_i, Uz_i, color='white', density=2.0,
                  linewidth=lw, arrowsize=1.0, zorder=1)

    th_bnd = np.linspace(0, 2*np.pi, 400)
    ax.plot(Rm * np.cos(th_bnd), Rm * np.sin(th_bnd), 'w-', lw=2)
    ax.plot(Rp * np.cos(th_bnd), Rp * np.sin(th_bnd), 'w-', lw=2)
    ax.set_xlim(-Rp - 0.08, Rp + 0.08)
    ax.set_ylim(-Rp - 0.08, Rp + 0.08)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('z', fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.7, label=r'$|\mathbf{u}|$')

    # Mark zoom regions
    # CMB zoom: theta ~ 45 deg
    zc_cmb = (Rm * np.sin(pi/4), Rm * np.cos(pi/4))
    zw = 0.12
    ax.plot([zc_cmb[0]-zw, zc_cmb[0]+zw, zc_cmb[0]+zw, zc_cmb[0]-zw, zc_cmb[0]-zw],
            [zc_cmb[1]-zw, zc_cmb[1]-zw, zc_cmb[1]+zw, zc_cmb[1]+zw, zc_cmb[1]-zw],
            'c-', lw=2, zorder=5)
    # Surface zoom: theta ~ 135 deg
    zc_surf = (Rp * np.sin(3*pi/4), Rp * np.cos(3*pi/4))
    zw_s = 0.12
    ax.plot([zc_surf[0]-zw_s, zc_surf[0]+zw_s, zc_surf[0]+zw_s, zc_surf[0]-zw_s, zc_surf[0]-zw_s],
            [zc_surf[1]-zw_s, zc_surf[1]-zw_s, zc_surf[1]+zw_s, zc_surf[1]+zw_s, zc_surf[1]-zw_s],
            'm-', lw=2, zorder=5)

# Row 0, col 2: difference
ax = fig.add_subplot(gs[0, 2])
ur_ns_i = ur_ns(Ri, Ti, Phi_i)
uth_ns_i = uth_ns(Ri, Ti, Phi_i)
ur_fs_i = ur_fs(Ri, Ti, Phi_i)
uth_fs_i = uth_fs(Ri, Ti, Phi_i)
vmag_ns_i = np.sqrt(ur_ns_i**2 + uth_ns_i**2)
vmag_fs_i = np.sqrt(ur_fs_i**2 + uth_fs_i**2)
diff = vmag_fs_i - vmag_ns_i
diff[mask] = np.nan
vabs = np.nanmax(np.abs(diff))
im = ax.pcolormesh(Xi, Zi, diff, cmap='RdBu_r', shading='auto',
                   vmin=-vabs, vmax=vabs, zorder=0)
th_bnd = np.linspace(0, 2*np.pi, 400)
ax.plot(Rm * np.cos(th_bnd), Rm * np.sin(th_bnd), 'k-', lw=2)
ax.plot(Rp * np.cos(th_bnd), Rp * np.sin(th_bnd), 'k-', lw=2)
ax.set_xlim(-Rp - 0.08, Rp + 0.08)
ax.set_ylim(-Rp - 0.08, Rp + 0.08)
ax.set_aspect('equal')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('z', fontsize=12)
ax.set_title(r'Difference $|\mathbf{u}|_\mathrm{fs} - |\mathbf{u}|_\mathrm{ns}$', fontsize=13)
fig.colorbar(im, ax=ax, shrink=0.7, label=r'$\Delta|\mathbf{u}|$')


# =========================================================================
# Helper: streamline zoom panel
def plot_stream_zoom(ax, ur_fn, uth_fn, cx, cz, hw, title):
    nf = 400
    xf = np.linspace(cx - hw, cx + hw, nf)
    zf = np.linspace(cz - hw, cz + hw, nf)
    Xf, Zf = np.meshgrid(xf, zf)
    Rf = np.sqrt(Xf**2 + Zf**2)
    Tf = np.arccos(np.clip(Zf / np.where(Rf > 1e-14, Rf, 1e-14), -1, 1))

    ur_f = ur_fn(Rf, Tf, phi0)
    uth_f = uth_fn(Rf, Tf, phi0)
    sign_x = np.where(Xf >= 0, 1.0, -1.0)
    Ux_f = sign_x * (ur_f * np.sin(Tf) + uth_f * np.cos(Tf))
    Uz_f = ur_f * np.cos(Tf) - uth_f * np.sin(Tf)
    vmag_f = np.sqrt(Ux_f**2 + Uz_f**2)

    mf = (Rf < Rm * 0.99) | (Rf > Rp * 1.01)
    Ux_f[mf] = np.nan
    Uz_f[mf] = np.nan
    vmag_f[mf] = np.nan

    ax.pcolormesh(Xf, Zf, vmag_f, cmap='inferno', shading='auto',
                  vmin=0, vmax=vmax_global, zorder=0)

    speed_f = np.sqrt(np.nan_to_num(Ux_f)**2 + np.nan_to_num(Uz_f)**2)
    lw = 2.5 * speed_f / (vmax_global + 1e-20)
    ax.streamplot(xf, zf, Ux_f, Uz_f, color='white', density=1.8,
                  linewidth=lw, arrowsize=0.8, zorder=1)

    th_bnd = np.linspace(0, 2*np.pi, 800)
    ax.plot(Rm * np.cos(th_bnd), Rm * np.sin(th_bnd), 'w-', lw=2.5)
    ax.plot(Rp * np.cos(th_bnd), Rp * np.sin(th_bnd), 'w-', lw=2.5)
    ax.set_xlim(cx - hw, cx + hw)
    ax.set_ylim(cz - hw, cz + hw)
    ax.set_aspect('equal')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.7, label=r'$|\mathbf{u}|$')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('z', fontsize=11)
    ax.set_title(title, fontsize=12)


# --- Row 1: Quiver zoom near CMB (theta ~ 45 deg) ---
cmb_cx = Rm * np.sin(pi/4)
cmb_cz = Rm * np.cos(pi/4)
hw_cmb = 0.12

plot_stream_zoom(fig.add_subplot(gs[1, 0]), ur_ns, uth_ns,
                 cmb_cx, cmb_cz, hw_cmb, 'Zoom CMB: ZeroSlip')
plot_stream_zoom(fig.add_subplot(gs[1, 1]), ur_fs, uth_fs,
                 cmb_cx, cmb_cz, hw_cmb, 'Zoom CMB: FreeZeroSlip')

# Row 1, col 2: radial profile of |u| at theta=pi/4
ax = fig.add_subplot(gs[1, 2])
r_prof = np.linspace(Rm, Rm + 0.3, 500)
th_prof = pi / 4
vmag_ns_prof = np.sqrt(ur_ns(r_prof, th_prof, phi0)**2 + uth_ns(r_prof, th_prof, phi0)**2)
vmag_fs_prof = np.sqrt(ur_fs(r_prof, th_prof, phi0)**2 + uth_fs(r_prof, th_prof, phi0)**2)
ax.plot(r_prof, vmag_ns_prof, 'b-', lw=2.5, label='ZeroSlip')
ax.plot(r_prof, vmag_fs_prof, 'r--', lw=2.5, label='FreeZeroSlip')
ax.axvline(Rm, color='gray', ls=':', lw=1.5, label='CMB')
ax.set_xlabel('r', fontsize=12)
ax.set_ylabel(r'$|\mathbf{u}|$', fontsize=12)
ax.set_title(r'Radial profile near CMB ($\theta=45°$, $\phi=0$)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# --- Row 2: Quiver zoom near surface (theta ~ 135 deg) ---
surf_cx = Rp * np.sin(3*pi/4)
surf_cz = Rp * np.cos(3*pi/4)
hw_surf = 0.12

plot_stream_zoom(fig.add_subplot(gs[2, 0]), ur_ns, uth_ns,
                 surf_cx, surf_cz, hw_surf, 'Zoom surface: ZeroSlip')
plot_stream_zoom(fig.add_subplot(gs[2, 1]), ur_fs, uth_fs,
                 surf_cx, surf_cz, hw_surf, 'Zoom surface: FreeZeroSlip')

# Row 2, col 2: radial profile of |u| near surface
ax = fig.add_subplot(gs[2, 2])
r_prof2 = np.linspace(Rp - 0.3, Rp, 500)
vmag_ns_prof2 = np.sqrt(ur_ns(r_prof2, th_prof, phi0)**2 + uth_ns(r_prof2, th_prof, phi0)**2)
vmag_fs_prof2 = np.sqrt(ur_fs(r_prof2, th_prof, phi0)**2 + uth_fs(r_prof2, th_prof, phi0)**2)
ax.plot(r_prof2, vmag_ns_prof2, 'b-', lw=2.5, label='ZeroSlip')
ax.plot(r_prof2, vmag_fs_prof2, 'r--', lw=2.5, label='FreeZeroSlip')
ax.axvline(Rp, color='gray', ls=':', lw=1.5, label='Surface')
ax.set_xlabel('r', fontsize=12)
ax.set_ylabel(r'$|\mathbf{u}|$', fontsize=12)
ax.set_title(r'Radial profile near surface ($\theta=45°$, $\phi=0$)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# --- Row 3: Full radial profiles for all velocity components ---
r_full = np.linspace(Rm, Rp, 500)
th_prof = pi / 4

components = [
    (r'$u_r$', ur_ns, ur_fs),
    (r'$u_\theta$', uth_ns, uth_fs),
    (r'$|\mathbf{u}|$', None, None),
]

for col, (label, fn_ns, fn_fs) in enumerate(components):
    ax = fig.add_subplot(gs[3, col])
    if fn_ns is not None:
        vals_ns = fn_ns(r_full, th_prof, phi0)
        vals_fs = fn_fs(r_full, th_prof, phi0)
    else:
        vals_ns = np.sqrt(ur_ns(r_full, th_prof, phi0)**2 + uth_ns(r_full, th_prof, phi0)**2)
        vals_fs = np.sqrt(ur_fs(r_full, th_prof, phi0)**2 + uth_fs(r_full, th_prof, phi0)**2)
    ax.plot(r_full, vals_ns, 'b-', lw=2.5, label='ZeroSlip')
    ax.plot(r_full, vals_fs, 'r--', lw=2.5, label='FreeZeroSlip')
    ax.axvline(Rm, color='gray', ls=':', lw=1, alpha=0.6)
    ax.axvline(Rp, color='gray', ls=':', lw=1, alpha=0.6)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('r', fontsize=12)
    ax.set_ylabel(label, fontsize=13)
    ax.set_title(f'{label} radial profile ' + r'($\theta=45°$, $\phi=0$)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

fig.suptitle('Assess Stokes solutions: ZeroSlip vs FreeZeroSlip (l=2, m=2, k=2)\n'
             r'Meridional cross-section at $\phi=0$, $R_m=0.5$, $R_p=1.0$, $\nu=1$, $g=1$',
             fontsize=15, y=1.0)
plt.savefig('crosscut_streamlines.png', dpi=150, bbox_inches='tight')
print('Saved crosscut_streamlines.png')
