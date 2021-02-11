from typing import List, Optional, Union, Sequence, Tuple, Any, Dict

import os
import math
from pathlib import Path

import numpy as np
import scipy.optimize as sciopt

from bag.core import BagProject
from bag.io.file import read_yaml
from bag.io.sim_data import load_sim_file
from bag.math.interpolate import interpolate_grid
from bag.math.dfun import VectorDiffFunction, DiffFunction
from bag.simulation.core import DesignSpecs
from bag.util.immutable import ImmutableList

# TODO: the following code requires further testing for full verification.
# This code was ported from the original BAG2 version


# Currently, DesignManager is not used for MOS characterization.
# As a result, this modified DesignSpecs class is a hack to get MOSDiscreteDB running
# without having to implement significant changes to both characterization and querying code
class MOSCharSpecs(DesignSpecs):
    """A class that parses the characterization specification file."""

    def __init__(self, spec_file: str, spec_dict: Optional[Dict[str, Any]] = None) -> None:
        if spec_dict:
            self._specs = spec_dict
            self._root_dir: Path = Path(self._specs['root_dir']).resolve()
        elif spec_file:
            spec_path = Path(spec_file).resolve()
            if spec_path.is_file():
                self._specs = read_yaml(spec_path)
                self._root_dir: Path = Path(self._specs['root_dir']).resolve()
            elif spec_path.is_dir():
                self._root_dir: Path = spec_path
                self._specs = read_yaml(self._root_dir / 'specs.yaml')
            else:
                raise ValueError(f'{spec_path} is neither data directory or specification file.')
        else:
            raise ValueError('spec_file is empty.')

        self._swp_var_list: ImmutableList[str] = ImmutableList(
            sorted(self._specs['sweep_params'].keys()))
        self._sweep_params = self._specs['sweep_params']

    @property
    def dsn_basename(self) -> str:
        return self._specs['impl_cell']  # FIXME


class MOSDBDiscrete:
    """Transistor small signal parameters database with discrete width choices.

    This class provides useful query/optimization methods and ways to store/retrieve
    data.

    Parameters
    ----------
    spec_list : List[str]
        list of specification file locations corresponding to widths.
    interp_method : str
        interpolation method.
    meas_type : str
        transistor characterization measurement type.
    vgs_res : float
        vgs resolution used when computing vgs from vstar.
    is_schematic : bool
        True if this is working with schematic simulation data.
    width_var : str
        the width variable name.
    """

    def __init__(self, prj: BagProject, spec_list: List[str], interp_method: str = 'spline',
                 vgs_res: float = 5e-3, width_var: str = 'w') -> None:

        self._width_res = prj.tech_info.tech_params['mos']['width_resolution']
        self._sim_envs = None
        self._ss_swp_names = None
        self._dsn_info_list: List[MOSCharSpecs] = []
        self._ss_list = []
        self._ss_outputs = None
        self._width_list = []
        self._vgs_res = vgs_res

        for spec in spec_list:
            dsn_info = MOSCharSpecs(spec)
            cur_width = dsn_info.specs['dut_params'][width_var]
            cur_width = int(round(cur_width / self._width_res))
            self._width_list.append(cur_width)

            # error checking
            if 'w' in dsn_info.swp_var_list:
                raise ValueError('MOSDBDiscrete assumes transistor width is not swept.')

            ss_fun_table = {}
            for dsn_name in dsn_info.dsn_name_iter():
                meas_dir = dsn_info.root_dir / dsn_info.specs['meas_name']  # FIXME
                ss_dict = load_sim_file(os.path.join(str(meas_dir), 'ss_params.hdf5'))

                cur_corners = ss_dict['corner'].tolist()
                cur_ss_swp_names = ss_dict['sweep_params']['ibias'][1:]
                if self._sim_envs is None:
                    # assign attributes for the first time
                    self._sim_envs = cur_corners
                    self._ss_swp_names = cur_ss_swp_names
                elif self._sim_envs != cur_corners:
                    raise ValueError('Simulation environments mismatch between given specs.')
                elif self._ss_swp_names != cur_ss_swp_names:
                    raise ValueError('signal-signal parameter sweep names mismatch.')

                cur_fun_dict = self._make_ss_functions(ss_dict, cur_corners, cur_ss_swp_names, interp_method)

                if self._ss_outputs is None:
                    self._ss_outputs = sorted(cur_fun_dict.keys())

                ss_fun_table[dsn_name] = cur_fun_dict

            self._dsn_info_list.append(dsn_info)
            self._ss_list.append(ss_fun_table)

        self._env_list = self._sim_envs
        self._cur_idx = 0
        self._dsn_params = dict(w=self._width_list[0] * self._width_res)

    @classmethod
    def _make_ss_functions(cls, ss_dict, corners, swp_names, interp_method):
        scale_list = []
        for name in swp_names:
            cur_xvec = ss_dict[name]
            scale_list.append((cur_xvec[0], cur_xvec[1] - cur_xvec[0]))

        fun_table = {}
        corner_sort_arg = np.argsort(corners)  # type: Sequence[int]
        for key in ss_dict['sweep_params'].keys():
            arr = ss_dict[key]
            fun_list = []
            for idx in corner_sort_arg:
                fun_list.append(interpolate_grid(scale_list, arr[idx, ...], method=interp_method,
                                                 extrapolate=True, delta=1e-5))
            fun_table[key] = fun_list

        # add derived parameters
        cgdl = fun_table['cgd']
        cgsl = fun_table['cgs']
        cgbl = fun_table['cgb']
        cdsl = fun_table['cds']
        cdbl = fun_table['cdb']
        csbl = fun_table['csb']
        gml = fun_table['gm']
        ibiasl = fun_table['ibias']
        fun_table['cgg'] = [cgd + cgs + cgb for (cgd, cgs, cgb) in zip(cgdl, cgsl, cgbl)]
        fun_table['cdd'] = [cgd + cds + cdb for (cgd, cds, cdb) in zip(cgdl, cdsl, cdbl)]
        fun_table['css'] = [cgs + cds + csb for (cgs, cds, csb) in zip(cgsl, cdsl, csbl)]
        fun_table['vstar'] = [2 * ibias / gm for (gm, ibias) in zip(gml, ibiasl)]

        return fun_table

    @property
    def width_list(self) -> List[Union[float, int]]:
        """Returns the list of widths in this database."""
        return [w * self._width_res for w in self._width_list]

    @property
    def env_list(self) -> List[str]:
        """The list of simulation environments to consider."""
        return self._env_list

    @env_list.setter
    def env_list(self, new_env_list: List[str]):
        """Sets the list of simulation environments to consider."""
        self._env_list = new_env_list

    @property
    def dsn_params(self) -> Sequence[str]:
        """List of design parameters."""
        return self._dsn_info_list[self._cur_idx].swp_var_list

    def get_dsn_param_values(self, var) -> List[Any]:
        """Returns a list of valid design parameter values."""
        return self._dsn_info_list[self._cur_idx].get_swp_values(var)

    def set_dsn_params(self, **kwargs):
        """Set the design parameters for which this database will query for."""
        self._dsn_params.update(kwargs)
        w_unit = int(round(self._dsn_params['w'] / self._width_res))
        self._cur_idx = self._width_list.index(w_unit)

    def _get_dsn_name(self, **kwargs) -> str:
        if kwargs:
            self.set_dsn_params(**kwargs)

        combo_list = tuple(self._dsn_params[var] for var in self.dsn_params)
        dsn_name = self._dsn_info_list[self._cur_idx].get_design_name(combo_list)
        if dsn_name not in self._ss_list[self._cur_idx]:
            raise ValueError('Unknown design name: %s.  Did you set design parameters?' % dsn_name)

        return dsn_name

    def get_function_list(self, name, **kwargs) -> List[DiffFunction]:
        """Returns a list of functions, one for each simulation environment, for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        **kwargs :
            design parameter values.

        Returns
        -------
        output : Union[RegGridInterpVectorFunction, RegGridInterpFunction]
            the output vector function.
        """
        dsn_name = self._get_dsn_name(**kwargs)
        cur_dict = self._ss_list[self._cur_idx][dsn_name]
        fun_list = []
        for env in self.env_list:
            try:
                env_idx = self._sim_envs.index(env)
            except ValueError:
                raise ValueError('environment %s not found.' % env)

            fun_list.append(cur_dict[name][env_idx])
        return fun_list

    def get_function(self, name, env='', **kwargs) -> Union[VectorDiffFunction, DiffFunction]:
        """Returns a function for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        env : str
            if not empty, we will return function for just the given simulation environment.
        **kwargs :
            design parameter values.

        Returns
        -------
        output : Union[RegGridInterpVectorFunction, RegGridInterpFunction]
            the output vector function.
        """
        if not env and len(self.env_list) == 1:
            env = self.env_list[0]

        if not env:
            return VectorDiffFunction(self.get_function_list(name, **kwargs))
        else:
            dsn_name = self._get_dsn_name(**kwargs)
            cur_dict = self._ss_list[self._cur_idx][dsn_name]
            try:
                env_idx = self._sim_envs.index(env)
            except ValueError:
                raise ValueError('environment %s not found.' % env)

            return cur_dict[name][env_idx]

    def get_fun_sweep_params(self, **kwargs) -> Tuple[List[str], List[Tuple[float, float]]]:
        """Returns interpolation function sweep parameter names and values.

        Parameters
        ----------
        **kwargs :
            design parameter values.

        Returns
        -------
        sweep_params : List[str]
            list of parameter names.
        sweep_range : List[Tuple[float, float]]
            list of parameter range
        """
        dsn_name = self._get_dsn_name(**kwargs)
        sample_fun = self._ss_list[self._cur_idx][dsn_name]['gm'][0]

        return self._ss_swp_names, sample_fun.input_ranges

    def get_fun_arg(self, vgs: Optional[float] = None, vds: Optional[float] = None, vbs: float = 0.0,
                    vstar: Optional[float] = None, env: str = '') -> np.ndarray:
        """Compute argument for small signal parameter functions for the given bias point.

        Either one of vgs and vstar must be specified.  If vds is not specified, we set vds = vgs.
        If vbs is not specified, we set vbs = 0.

        You can specify vstar only if we only consider one simulation environment.

        Parameters
        ----------
        vgs : Optional[float]
            gate-to-source voltage.  For PMOS this is negative.
        vds : Optional[float]
            drain-to-source voltage.  For PMOS this is negative.
        vbs : float
            body-to-source voltage.  For NMOS this is negative.
        vstar : Optional[float]
            vstar, or 2 * id / gm.  This is always positive.
        env : str
            If not empty, will return results for this simulation environment only.

        Returns
        -------
        arg : np.ndarray
            the argument to pass to small signal parameter functions.
        """
        bias_info = self._get_bias_point_info(vgs=vgs, vds=vds, vbs=vbs, vstar=vstar, env=env)
        return np.array([bias_info[key] for key in self._ss_swp_names])

    def _get_bias_point_info(self, vgs: Optional[float] = None, vds: Optional[float] = None, vbs: float = 0.0,
                             vstar: Optional[float] = None, env: str = '') -> Dict[str, float]:
        """Compute bias point dictionary from given specs."""
        if vgs is None:
            if vstar is None:
                raise ValueError('At least one of vgs or vstar must be defined.')
            # check we only have one environment
            if not env:
                if len(self.env_list) > 1:
                    raise ValueError('Cannot compute bias point from vstar if we have more than one simulation'
                                     'environment.')
                env = self.env_list[0]

            # compute vgs from vstar spec
            # first, get vgs bounds
            fun_vstar = self.get_function('vstar', env=env)
            vgs_idx = self.get_fun_arg_index('vgs')
            vgs_min, vgs_max = fun_vstar.get_input_range(vgs_idx)
            if vds is None:
                vds_idx = self.get_fun_arg_index('vds')
                vds_min, vds_max = fun_vstar.get_input_range(vds_idx)
                vgs_min = max(vgs_min, vds_min)
                vgs_max = min(vgs_max, vds_max)

            # define vstar function.  Can do batch input.
            ndim = len(self._ss_swp_names)
            op_dict = dict(vds=vds, vbs=vbs)

            def fzero(vtest):
                vstar_arg = np.zeros([np.size(vtest), ndim])
                for idx, key in enumerate(self._ss_swp_names):
                    if key == 'vgs' or key == 'vds' and op_dict['vds'] is None:
                        vstar_arg[:, idx] = vtest
                    else:
                        vstar_arg[:, idx] = op_dict[key]
                return fun_vstar(vstar_arg) - vstar

            # do a coarse sweep to find maximum and minimum vstar.
            # NOTE: we do a coarse sweep because for some technologies, if we
            # are near or below threshold, vstar actually may not be monotonic
            # function of vgs.
            num_pts = int(math.ceil((vgs_max - vgs_min) / self._vgs_res)) + 1
            vgs_vec = np.linspace(vgs_min, vgs_max, num_pts)
            vstar_diff = fzero(vgs_vec)

            if abs(vgs_max) >= abs(vgs_min):
                # NMOS.  We want to find the last vgs with smaller vstar
                idx1 = num_pts - 1 - np.argmax(vstar_diff[::-1] < 0)
                if vstar_diff[idx1] > 0:
                    raise ValueError('vstar = %.4g unachieveable; min vstar = %.4g' %
                                     (vstar, np.min(vstar_diff + vstar)))
                idx2 = idx1 + 1
                if idx2 >= num_pts or vstar_diff[idx2] < 0:
                    raise ValueError('vstar = %.4g unachieveable; max vstar = %.4g' %
                                     (vstar, np.max(vstar_diff + vstar)))
            else:
                # PMOS, we want to find first vgs with smaller vstar
                idx2 = np.argmax(vstar_diff <= 0)
                if vstar_diff[idx2] > 0:
                    raise ValueError('vstar = %.4g unachieveable; min vstar = %.4g' %
                                     (vstar, np.min(vstar_diff + vstar)))
                idx1 = idx2 - 1
                if idx1 < 0 or vstar_diff[idx1] < 0:
                    raise ValueError('vstar = %.4g unachieveable; max vstar = %.4g' %
                                     (vstar, np.max(vstar_diff + vstar)))

            vgs = sciopt.brentq(fzero, vgs_vec[idx1], vgs_vec[idx2])

        if vds is None:
            # set vds if not specified
            vds = vgs

        return dict(vgs=vgs, vds=vds, vbs=vbs)

    def get_fun_arg_index(self, name: str) -> int:
        """Returns the function input argument index for the given variable

        Parameters
        ----------
        name : str
            one of vgs, vds, or vbs.

        Returns
        -------
        idx : int
            index of the given argument.
        """
        return self._ss_swp_names.index(name)

    def query(self, vgs: Optional[float] = None, vds: Optional[float] = None, vbs: float = 0.0,
              vstar: Optional[float] = None, env: str = '') -> Dict[str, np.ndarray]:
        """Query the database for the values associated with the given parameters.

        Either one of vgs and vstar must be specified.  If vds is not specified, we set vds = vgs.
        If vbs is not specified, we set vbs = 0.

        Parameters
        ----------
        vgs : Optional[float]
            gate-to-source voltage.  For PMOS this is negative.
        vds : Optional[float]
            drain-to-source voltage.  For PMOS this is negative.
        vbs : float
            body-to-source voltage.  For NMOS this is negative.
        vstar : Optional[float]
            vstar, or 2 * id / gm.  This is always positive.
        env : str
            If not empty, will return results for this simulation environment only.

        Returns
        -------
        results : Dict[str, np.ndarray]
            the characterization results.
        """
        bias_info = self._get_bias_point_info(vgs=vgs, vds=vds, vbs=vbs, vstar=vstar, env=env)
        fun_arg = np.array([bias_info[key] for key in self._ss_swp_names])
        results = {name: self.get_function(name, env=env)(fun_arg) for name in self._ss_outputs}

        # add bias point information to result
        results.update(bias_info)
        return results
