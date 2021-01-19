from typing import Any, Union, Sequence, Tuple, Optional, Mapping, Iterable, List, Set, Dict, Type

import abc
import numpy as np

from bag.simulation.data import SimData, AnalysisType
from bag.design.module import Module
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimNetlistInfo, netlist_info_from_dict
from bag3_liberty.data import parse_cdba_name, BusRange

from ..tran.analog import AnalogTranTB

from ...schematic.digital_tb_tran import bag3_testbenches__digital_tb_tran
from ...schematic.analog_tb_tran import bag3_testbenches__analog_tb_tran


class PNoiseTB(AnalogTranTB):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        pnoise_options: Mapping[str, Any] = specs.get('pnoise_options', {})
        trigger_dir = pnoise_options.get('trigger_dir', 'rise')
        probe_thres = pnoise_options.get('probe_thres', 'v_VDD/2')
        probe_pin = pnoise_options.get('probe_pins', '[outp outn]')
        pnoise_options['events']['pm'] += f' trigger={probe_pin} target={probe_pin} triggerthresh={probe_thres} ' \
                                          f'triggerdir={trigger_dir}'
        pss_dict = dict(
            type='PSS',
            fund='1/t_per',
            options=dict(
                harms=pnoise_options.get('harms', 100),
                errpreset=pnoise_options.get('errpreset', 'conservative'),
                tstab=pnoise_options.get('tstab', 0),
                autosteady=pnoise_options.get('autosteady', 'yes'),
                maxacfreq=1e10,
            ),
            save_outputs=self.save_outputs,
        )
        pnoise_dict = dict(
            type='PNOISE',
            start=1,
            stop='0.5/t_per',
            options=dict(
                pnoisemethod=pnoise_options.get('pnoisemethod', 'fullspectrum'),
                noisetype=pnoise_options.get('noisetype', 'sampled'),
                saveallsidebands='yes',
                lorentzian='yes',
            ),
            events=pnoise_options['events'],
            save_outputs=self.save_outputs,
        )
        pac_dict = dict(
            type='PAC',
            p_port=pnoise_options.get('p_port', 'outp'),
            n_port=pnoise_options.get('n_port', 'outn'),
            start=pnoise_options.get('ac_start', 1),
            stop=pnoise_options.get('ac_stop', 100e9),
            options=dict(
                crossingdirection=trigger_dir,
                thresholdvalue=probe_thres,
                maxsideband=0,
                sweeptype='relative',
                ptvtype='sampled'
            ),
            save_outputs=self.save_outputs,
        )

        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [pss_dict, pnoise_dict, pac_dict]
        return netlist_info_from_dict(sim_setup)
