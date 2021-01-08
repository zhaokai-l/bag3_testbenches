"""This package contains measurement class for transistors."""

from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any, List, Mapping, Sequence, Union, Type, cast

import math
from pathlib import Path

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt

from bag.design.module import Module
from bag.io.file import write_yaml
from bag.io.sim_data import save_sim_results, load_sim_file
from bag.math.interpolate import LinearInterpolator
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.core import TestbenchManager
from bag.simulation.data import AnalysisType, SimNetlistInfo, SimData, netlist_info_from_dict
from bag.simulation.measure import MeasurementManager, MeasInfo

from ...schematic.mos_tb_ibias import bag3_testbenches__mos_tb_ibias
from ...schematic.mos_tb_sp import bag3_testbenches__mos_tb_sp
from ...schematic.mos_tb_noise import bag3_testbenches__mos_tb_noise


class MOSIdTB(TestbenchManager):
    """This class sets up the transistor drain current measurement testbench.
    """

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        return bag3_testbenches__mos_tb_ibias

    def get_netlist_info(self) -> SimNetlistInfo:
        dc_dict = dict(type='DC')

        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [dc_dict]
        return netlist_info_from_dict(sim_setup)

    def pre_setup(self, sch_params: Optional[Mapping[str, Any]]):
        self.sim_params['vs'] = 0
        vgs_max = self.specs['vgs_max']
        vgs_min = self.specs.get('vgs_min', 0)
        vgs_num = self.specs['vgs_num']
        if self.specs['is_nmos']:
            vgs_start, vgs_stop = vgs_min, vgs_max
        else:
            vgs_start, vgs_stop = -vgs_max, -vgs_min

        self.set_swp_info([
            ('vgs', dict(type='LINEAR', start=vgs_start, stop=vgs_stop, num=vgs_num))
        ])

        return super().pre_setup(sch_params)

    def get_vgs_range(self, data: SimData):
        ibias_min_seg = self.specs['ibias_min_seg']
        ibias_max_seg = self.specs['ibias_max_seg']
        vgs_res = self.specs['vgs_resolution']
        seg = self.specs['seg']
        is_nmos = self.specs['is_nmos']

        # invert NMOS ibias sign
        ibias_sgn = -1.0 if is_nmos else 1.0

        vgs = data['vgs'][0, :]  # remove corner index
        ibias_key = 'VD:p'
        ibias = data[ibias_key] * ibias_sgn

        # assume first sweep parameter is corner, second sweep parameter is vgs
        try:
            corner_idx = data.sweep_params.index('corner')
            ivec_max = np.amax(ibias, corner_idx)
            ivec_min = np.amin(ibias, corner_idx)
        except ValueError:
            ivec_max = ivec_min = ibias

        vgs1 = self._get_best_crossing(vgs, ivec_max, ibias_min_seg * seg)
        vgs2 = self._get_best_crossing(vgs, ivec_min, ibias_max_seg * seg)

        vgs_min = min(vgs1, vgs2)
        vgs_max = max(vgs1, vgs2)

        vgs_min = math.floor(vgs_min / vgs_res) * vgs_res
        vgs_max = math.ceil(vgs_max / vgs_res) * vgs_res

        return vgs_min, vgs_max

    @classmethod
    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop)
        except ValueError:
            # avoid no solution
            if abs(fzero(xstart)) < abs(fzero(xstop)):
                return xstart
            return xstop


class MOSSPTB(TestbenchManager):
    """This class sets up the transistor S parameter measurement testbench.
    """

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        return bag3_testbenches__mos_tb_sp

    def get_netlist_info(self) -> SimNetlistInfo:
        dc_dict = dict(type='DC')
        sp_dict = dict(type='SP',
                       freq=self.specs['sp_freq'],
                       ports=['PORTG', 'PORTD', 'PORTS'],
                       param_type='Y')

        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [dc_dict, sp_dict]
        return netlist_info_from_dict(sim_setup)

    def pre_setup(self, sch_params: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        is_nmos = self.specs['is_nmos']
        vbs_val = self.specs['vbs']
        vds_min = self.specs['vds_min']
        vds_max = self.specs['vds_max']
        vds_num = self.specs['vds_num']
        vgs_num = self.specs['vgs_num']
        adjust_vbs_sign = self.specs.get('adjust_vbs_sign', True)
        vgs_start, vgs_stop = self.specs['vgs_range']

        swp_info = []
        # Add VGS sweep
        swp_info.append(('vgs', dict(type='LINEAR', start=vgs_start, stop=vgs_stop, num=vgs_num)))

        # handle VBS sign and set parameters.
        if isinstance(vbs_val, list):
            if adjust_vbs_sign:
                print('adjusting vbs sign')
                if is_nmos:
                    vbs_val = sorted((-abs(v) for v in vbs_val))
                else:
                    vbs_val = sorted((abs(v) for v in vbs_val))
            else:
                vbs_val = sorted(vbs_val)
            print('vbs values: {}'.format(vbs_val))
            swp_info.append(('vbs', dict(type='LIST', values=vbs_val)))
        else:
            if adjust_vbs_sign:
                print('adjusting vbs sign')
                if is_nmos:
                    vbs_val = -abs(vbs_val)
                else:
                    vbs_val = abs(vbs_val)
            print('vbs value: {:.4g}'.format(vbs_val))
            self.sim_params['vbs'] = vbs_val

        # handle VDS/VGS sign for nmos/pmos
        if is_nmos:
            self.sim_params['vb_dc'] = 0
            vds_start, vds_stop = vds_min, vds_max
        else:
            if vds_max > vds_min:
                print('vds_max = {:.4g} > {:.4g} = vds_min, flipping sign'.format(vds_max, vds_min))
                vds_start, vds_stop = -vds_max, -vds_min
            else:
                vds_start, vds_stop = vds_min, vds_max
            self.sim_params['vb_dc'] = abs(vgs_start)

        swp_info.append(('vds', dict(type='LINEAR', start=vds_start, stop=vds_stop, num=vds_num)))

        self.set_swp_info(swp_info)

        return super().pre_setup(sch_params)

    def get_ss_params(self, data: SimData) -> Dict[str, Any]:
        cfit_method = self.specs['cfit_method']
        sp_freq = self.specs['sp_freq']
        seg = self.specs['seg']
        is_nmos = self.specs['is_nmos']

        swp_vars = data.sweep_params

        data.open_analysis(AnalysisType.DC)

        # invert NMOS ibias sign
        ibias_sgn = -1.0 if is_nmos else 1.0
        ibias_key = 'VD:p'
        ibias = data[ibias_key] * ibias_sgn

        data.open_analysis(AnalysisType.SP)

        data_dict = data._cur_ana._data
        ss_dict = self.mos_y_to_ss(data_dict, sp_freq, seg, ibias, cfit_method=cfit_method)

        new_result = {}
        new_shape = list(data.data_shape)
        del new_shape[data.sweep_params.index('freq')]

        sweep_params = {}
        for key, val in ss_dict.items():
            new_result[key] = val.reshape(new_shape)
            sweep_params[key] = swp_vars
        new_result['corner'] = np.array(self.sim_envs)

        for var in swp_vars:
            if var == 'corner':
                continue
            new_result[var] = data_dict[var]
        new_result['sweep_params'] = sweep_params

        return new_result

    @classmethod
    def mos_y_to_ss(cls, sim_data: Dict[str, np.ndarray], char_freq: float, seg: int, ibias: np.ndarray,
                    cfit_method: str = 'average') -> Dict[str, np.ndarray]:
        """Convert transistor Y parameters to small-signal parameters.

        This function computes MOSFET small signal parameters from 3-port
        Y parameter measurements done on gate, drain and source, with body
        bias fixed.  This functions fits the Y parameter to a capcitor-only
        small signal model using least-mean-square error.

        Parameters
        ----------
        sim_data : Dict[str, np.ndarray]
            A dictionary of Y parameters values stored as complex numpy arrays.
        char_freq : float
            the frequency Y parameters are measured at.
        seg : int
            number of transistor fingers used for the Y parameter measurement.
        ibias : np.ndarray
            the DC bias current of the transistor.  Always positive.
        cfit_method : str
            method used to extract capacitance from Y parameters.  Currently
            supports 'average' or 'worst'

        Returns
        -------
        ss_dict : Dict[str, np.ndarray]
            A dictionary of small signal parameter values stored as numpy
            arrays.  These values are normalized to 1-finger transistor.
        """
        w = 2 * np.pi * char_freq

        gm = (sim_data['y21'].real - sim_data['y31'].real) / 2.0
        gds = (sim_data['y22'].real - sim_data['y32'].real) / 2.0
        gb = (sim_data['y33'].real - sim_data['y23'].real) / 2.0 - gm - gds

        cgd12 = -sim_data['y12'].imag / w
        cgd21 = -sim_data['y21'].imag / w
        cgs13 = -sim_data['y13'].imag / w
        cgs31 = -sim_data['y31'].imag / w
        cds23 = -sim_data['y23'].imag / w
        cds32 = -sim_data['y32'].imag / w
        cgg = sim_data['y11'].imag / w
        cdd = sim_data['y22'].imag / w
        css = sim_data['y33'].imag / w

        if cfit_method == 'average':
            cgd = (cgd12 + cgd21) / 2
            cgs = (cgs13 + cgs31) / 2
            cds = (cds23 + cds32) / 2
        elif cfit_method == 'worst':
            cgd = np.maximum(cgd12, cgd21)
            cgs = np.maximum(cgs13, cgs31)
            cds = np.maximum(cds23, cds32)
        else:
            raise ValueError('Unknown cfit_method = %s' % cfit_method)

        cgb = cgg - cgd - cgs
        cdb = cdd - cds - cgd
        csb = css - cgs - cds

        ibias = ibias / seg
        gm = gm / seg
        gds = gds / seg
        gb = gb / seg
        cgd = cgd / seg
        cgs = cgs / seg
        cds = cds / seg
        cgb = cgb / seg
        cdb = cdb / seg
        csb = csb / seg

        return dict(
            ibias=ibias,
            gm=gm,
            gds=gds,
            gb=gb,
            cgd=cgd,
            cgs=cgs,
            cds=cds,
            cgb=cgb,
            cdb=cdb,
            csb=csb,
        )


# TODO: needs to be "translated" to BAG3 and verified
class MOSNoiseTB(TestbenchManager):
    """This class sets up the transistor small-signal noise measurement testbench.
    """
    def __init__(self,
                 data_fname,  # type: str
                 tb_name,  # type: str
                 impl_lib,  # type: str
                 specs,  # type: Dict[str, Any]
                 sim_view_list,  # type: Sequence[Tuple[str, str]]
                 env_list,  # type: Sequence[str]
                 ):
        # type: (...) -> None
        TestbenchManager.__init__(self, data_fname, tb_name, impl_lib, specs, sim_view_list, env_list)

    def setup_testbench(self, tb):
        vbs_val = self.specs['vbs']
        vds_min = self.specs['vds_min']
        vds_max = self.specs['vds_max']
        vds_num = self.specs['vds_num']
        vgs_num = self.specs['vgs_num']
        freq_start = self.specs['freq_start']
        freq_stop = self.specs['freq_stop']
        num_per_dec = self.specs['num_per_dec']

        vgs_start, vgs_stop = self.specs['vgs_range']
        is_nmos = self.specs['is_nmos']

        # handle VBS sign and set parameters.
        if isinstance(vbs_val, list):
            if is_nmos:
                vbs_val = sorted((-abs(v) for v in vbs_val))
            else:
                vbs_val = sorted((abs(v) for v in vbs_val))
            tb.set_sweep_parameter('vbs', values=vbs_val)
        else:
            if is_nmos:
                vbs_val = -abs(vbs_val)
            else:
                vbs_val = abs(vbs_val)
            tb.set_parameter('vbs', vbs_val)

        tb.set_parameter('freq_start', freq_start)
        tb.set_parameter('freq_stop', freq_stop)
        tb.set_parameter('num_per_dec', num_per_dec)

        vgs_vals = np.linspace(vgs_start, vgs_stop, vgs_num + 1)
        # handle VDS/VGS sign for nmos/pmos
        if is_nmos:
            vds_vals = np.linspace(vds_min, vds_max, vds_num + 1)
            tb.set_sweep_parameter('vds', values=vds_vals)
            tb.set_sweep_parameter('vgs', values=vgs_vals)
            tb.set_parameter('vb_dc', 0)
        else:
            vds_vals = np.linspace(-vds_max, -vds_min, vds_num + 1)
            tb.set_sweep_parameter('vds', values=vds_vals)
            tb.set_sweep_parameter('vgs', values=vgs_vals)
            tb.set_parameter('vb_dc', abs(vgs_start))

    def get_integrated_noise(self, data, ss_data, temp, fstart, fstop, scale=1.0):
        seg = self.specs['seg']

        axis_names = ['corner', 'vbs', 'vds', 'vgs', 'freq']

        idn = data['idn']

        ss_swp_names = [name for name in axis_names[1:] if name in data]
        swp_corner = ('corner' in data)
        if not swp_corner:
            data = data.copy()
            data['corner'] = np.array([self.env_list[0]])
        corner_list = data['corner']
        log_freq = np.log(data['freq'])
        cur_points = [data[name] for name in ss_swp_names]
        cur_points[-1] = log_freq

        # rearrange array axis
        swp_vars = data['sweep_params']['idn']
        new_swp_vars = ['corner', ] + ss_swp_names[:-1]
        order = [swp_vars.index(name) for name in axis_names if name in swp_vars]
        idn = np.transpose(idn, axes=order)
        if not swp_corner:
            # add dimension that corresponds to corner
            idn = idn[np.newaxis, ...]

        # construct new SS parameter result dictionary
        fstart_log = np.log(fstart)
        fstop_log = np.log(fstop)

        # rearrange array axis
        idn = np.log(scale / seg * (idn ** 2))
        delta_list = [1e-6] * len(ss_swp_names)
        delta_list[-1] = 1e-3
        integ_noise_list = []
        for idx in range(len(corner_list)):
            noise_fun = LinearInterpolator(cur_points, idn[idx, ...], delta_list, extrapolate=True)
            integ_noise_list.append(noise_fun.integrate(fstart_log, fstop_log, axis=-1, logx=True, logy=True, raw=True))

        gamma = np.array(integ_noise_list) / (4.0 * 1.38e-23 * temp * ss_data['gm'] * (fstop - fstart))
        self.record_array(ss_data, data, gamma, 'gamma', new_swp_vars)
        return ss_data


class MOSCharSS(MeasurementManager):
    """This class measures small signal parameters of a transistor using Y parameter fitting.

    This measurement is perform as follows:

    1. First, given a user specified current density range, we perform a DC current measurement
       to find the range of vgs needed across corners to cover that range.
    2. Then, we run a S parameter simulation and record Y parameter values at various bias points.
    3. If user specify a noise testbench, a noise simulation will be run at the same bias points
       as S parameter simulation to characterize transistor noise.

    Parameters
    ----------
    data_dir : str
        Simulation data directory.
    meas_name : str
        measurement setup name.
    impl_lib : str
        implementation library name.
    specs : Dict[str, Any]
        the measurement specification dictionary.
    wrapper_lookup : Dict[str, str]
        the DUT wrapper cell name lookup table.
    sim_view_list : Sequence[Tuple[str, str]]
        simulation view list
    env_list : Sequence[str]
        simulation environments list.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tbm_cls_map = dict(
            ibias=MOSIdTB,
            sp=MOSSPTB,
            noise=MOSNoiseTB
        )

    @property
    def tbm_order(self) -> List[str]:
        """
        Returns a list of measurement manager names in the order which should be run
        Pulse response measurement should be run first to determine optimal input arrival time

        """
        return ['ibias', 'sp', 'noise']

    def commit(self):
        super().commit()

        self._sim_envs = self.specs['sim_envs']

        # Update which sub-measurements should be run
        self._run_tbm = {tbm_name: False for tbm_name in self.tbm_order}

        tbm_specs_shared = {k: v for k, v in self.specs.items() if k not in ['tbm_specs']}

        # Update each tbm specs, including parameters that are shared across all tbms
        for tbm_name, tbm_specs in self.specs['tbm_specs'].items():
            assert tbm_name in self.tbm_order
            self._run_tbm[tbm_name] = True
            self.specs['tbm_specs'][tbm_name] = tbm_specs_shared.copy()
            self.specs['tbm_specs'][tbm_name].update(tbm_specs)

    # MeasurementManager's async_measure_performance utilizes the following 3 functions
    # As async_measure_performance has been rewritten to better suit MOSCharSS behavior,
    # these methods are no longer needed.

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo):
        raise NotImplementedError

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance):
        raise NotImplementedError

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]):
        raise NotImplementedError

    def add_tbm(self, tbm_name: str) -> TestbenchManager:
        """
        Add/create a testbench manager

        Parameters
        ----------
        tbm_name : str
            name of testbench manager

        Returns
        -------
        Newly created testbench manager

        """
        assert tbm_name in self.tbm_dict
        tbm_cls = self.tbm_cls_map[tbm_name]
        self.tbm_dict[tbm_name] = cast(tbm_cls, self.make_tbm(tbm_cls, self.specs['tbm_specs'][tbm_name]))
        return self.tbm_dict[tbm_name]

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        """
        A coroutine that performs measurement.
        This will...
        1. Acquire every sweep point/configuration (get_swp_list)
        2. Run simulation for each sweep point (_run_sim)
        3. Aggregate results and return (aggregate_results)

        Parameters
        ----------
        name : str
            name of measurement
        sim_dir : Path
            simulation directory
        sim_db : SimulationDB
            the simulation database object
        dut : Optional[DesignInstance]
            the DUT to measure

        Returns
        -------
        results dictionary

        """
        assert len(self._sim_envs) > 0
        self.tbm_dict: Dict[str, TestbenchManager] = {k: None for k in self.tbm_order}

        res = {}

        for idx, tbm_name in enumerate(self.tbm_order):
            if not self._run_tbm[tbm_name]:
                continue

            tbm = self.add_tbm(tbm_name)
            sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir / tbm_name, dut, tbm, tb_params={})
            data = sim_results.data

            ss_fname = str(sim_dir / 'ss_params.hdf5')

            if tbm_name == 'ibias':
                tbm: MOSIdTB
                vgs_range = tbm.get_vgs_range(data)
                if self._run_tbm['sp']:
                    self.specs['tbm_specs']['sp']['vgs_range'] = vgs_range
                if self._run_tbm['noise']:
                    self.specs['tbm_specs']['noise']['vgs_range'] = vgs_range
                self.commit()
                res['vgs_range'] = vgs_range

            elif tbm_name == 'sp':
                tbm: MOSSPTB
                ss_params = tbm.get_ss_params(data)
                # save SS parameters
                save_sim_results(ss_params, ss_fname)

                res['ss_file'] = ss_fname

            elif tbm_name == 'noise':
                # TODO: needs to be verified
                tbm: MOSNoiseTB
                temp = self.specs['noise_temp_kelvin']
                fstart = self.specs['noise_integ_fstart']
                fstop = self.specs['noise_integ_fstop']
                scale = self.specs.get('noise_integ_scale', 1.0)

                ss_params = load_sim_file(ss_fname)
                ss_params = tbm.get_integrated_noise(data, ss_params, temp, fstart, fstop, scale=scale)
                save_sim_results(ss_params, ss_fname)

                res['ss_file'] = ss_fname

            else:
                raise ValueError(f"Unknown tbm name {tbm_name}")

        write_yaml(sim_dir / f'{name}.yaml', res)

        return res
