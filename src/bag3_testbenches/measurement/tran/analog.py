# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union, Sequence, Tuple, Optional, Mapping, Iterable, List, Set, Dict

# from itertools import chain

import numpy as np

# from pybag.core import get_cdba_name_bits

from bag.simulation.data import SimData, AnalysisType

from bag3_liberty.data import parse_cdba_name, BusRange

from ..data.tran import EdgeType, get_first_crossings
from .base import AnaTranTB


class AnalogTranTB(AnaTranTB):
    """A transient testbench with Analog stimuli.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def get_pin_supplies(cls, pin_name: str, pwr_domain: Mapping[str, Tuple[str, str]]
                         ) -> Tuple[str, str]:
        ans = pwr_domain.get(pin_name, None)
        if ans is None:
            basename, _ = parse_cdba_name(pin_name)
            return pwr_domain[basename]
        return ans

    def pre_setup(self, sch_params: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        """Set up waveform files."""
        if sch_params is None:
            return None

        specs = self.specs
        sup_values: Mapping[str, Union[float, Mapping[str, float]]] = specs['sup_values']
        dut_pins: Sequence[str] = specs['dut_pins']
        stimuli_list: Sequence[Mapping[str, Any]] = specs.get('stimuli_list', [])
        load_list: Sequence[Mapping[str, Any]] = specs.get('load_list', [])
        other_list: List[Mapping[str, str]] = specs.get('other_list', [])
        dut_conns_preset: Dict[str, str] = specs.get('dut_conns', dict())
        # set digital bits
        pin_values: Mapping[str, int] = specs.get('pin_values', {})

        if 'pwr_domain' not in self.specs.keys():
            self.specs['pwr_domain'] = {p_: ('VSS', 'VDD') for p_ in list(dut_pins)}

        src_list = []
        src_pins = set()
        self.get_stimuli_sources(stimuli_list, src_list, src_pins)
        self.get_bias_sources(sup_values, src_list, src_pins)
        self.get_loads(load_list, src_list)

        dut_conns, no_conns = self.get_dut_conns(dut_pins, src_pins, pin_values)
        dut_conns.update(dut_conns_preset)
        for pin, conn in dut_conns_preset.items():
            if pin in dut_conns:
                no_conns.remove(pin)
        return dict(
            dut_lib=sch_params.get('dut_lib', ''),
            dut_cell=sch_params.get('dut_cell', ''),
            dut_params=sch_params.get('dut_params', None),
            dut_conns=dut_conns,
            vbias_list=[],
            src_list=src_list,
            other_list=other_list,
        )

    def get_stimuli_sources(self, waveform_list: Iterable[Mapping[str, Any]],
                            src_list: List[Mapping[str, Any]], src_pins: Set[str]) -> None:
        specs = self.specs
        pwr_domain: Mapping[str, Tuple[str, str]] = specs['pwr_domain']
        skip_src: bool = specs.get('skip_src', False)

        for wave_params in waveform_list:
            pin: str = wave_params['pin']

            if pin in src_pins:
                if skip_src:
                    continue
                else:
                    raise ValueError(f'Cannot add pulse source on pin {pin}, already used.')

            gnd_name, pwr_name = self.get_pin_supplies(pin, pwr_domain)
            gnd_name = wave_params.get('gnd', gnd_name)
            src_type: str = wave_params['src_type']
            table: Mapping[str, Any] = wave_params['table']
            # for v, k in table:
            #     src_table.update(k=self.get_sim_param_string(v))
            src_pins.add(pin)
            src_list.append(dict(type=src_type, lib='analogLib', value=table,
                                 conns=dict(PLUS=pin, MINUS=gnd_name)))

    def get_loads(self, load_list: Iterable[Mapping[str, Any]],
                  src_load_list: List[Mapping[str, Any]]) -> None:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        for params in load_list:
            pin: str = params['pin']
            value: Union[float, str] = params['value']
            dev_type: str = params['type']
            gnd_name = self.get_pin_supplies(pin, pwr_domain)[1]
            gnd_name = params.get('gnd', gnd_name)
            src_load_list.append(dict(type=dev_type, lib='analogLib', value=value,
                                      conns=dict(PLUS=pin, MINUS=gnd_name)))

    def get_dut_conns(self, dut_pins: Iterable[str], src_pins: Set[str],
                      pin_values: Mapping[str, int]) -> Tuple[Dict[str, str], List[str]]:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        ans = {}
        no_conn = []
        for pin_name in dut_pins:
            pin_val: Optional[int] = pin_values.get(pin_name, None)
            basename, bus_range = parse_cdba_name(pin_name)
            if bus_range is None:
                # scalar pins
                if pin_name in src_pins or pin_val is None:
                    ans[pin_name] = pin_name
                else:
                    ans[pin_name] = self.get_pin_supplies(pin_name, pwr_domain)[pin_val]
                if pin_name not in src_pins and pin_val is None:
                    no_conn.append(pin_name)
            else:
                # bus pins
                if pin_val is None:
                    # no bias values specified
                    ans[pin_name] = pin_name
                    no_conn.append(pin_name)
                    # [no_conn.append(f"pin_name<{idx}>") for idx in range(max(bus_range.start, bus_range.stop))]
                else:
                    nlen = len(bus_range)
                    bin_str = bin(pin_val)[2:].zfill(nlen)
                    ans[pin_name] = self._bin_str_to_net(basename, bus_range, bin_str, pwr_domain,
                                                         src_pins)

        return ans, no_conn

    def calc_cross(self, data: SimData, out_name: str, out_edge: EdgeType,
                   t_start: Union[np.ndarray, float, str] = 0,
                   t_stop: Union[np.ndarray, float, str] = float('inf')) -> np.ndarray:
        thres_delay = 0.5

        specs = self.specs
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        out_0, out_1 = self.get_pin_supply_values(out_name, data)
        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        out_vec = data[out_name]

        # evaluate t_start/t_stop
        if isinstance(t_start, str) or isinstance(t_stop, str):
            calc = self.get_calculator(data)
            if isinstance(t_start, str):
                t_start = calc.eval(t_start)
            if isinstance(t_stop, str):
                t_stop = calc.eval(t_stop)

        vth_out = (out_1 - out_0) * thres_delay + out_0
        out_c = get_first_crossings(tvec, out_vec, vth_out, etype=out_edge, start=t_start,
                                    stop=t_stop, rtol=rtol, atol=atol)
        return out_c

    def calc_delay(self, data: SimData, in_name: str, out_name: str, in_edge: EdgeType,
                   out_edge: EdgeType, diff_in: bool, diff_out: bool, t_start: Union[np.ndarray, float, str] = 0,
                   t_stop: Union[np.ndarray, float, str] = float('inf'),  tvec: np.ndarray = np.array([]),
                   data_in: np.ndarray = np.array([]), data_out: np.ndarray = np.array([])) -> np.ndarray:
        thres_delay = 0.5

        specs = self.specs
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        data.open_analysis(AnalysisType.TRAN)
        tvec = tvec if tvec.any() else data['time']
        if data_in.any():
            in_vec = data_in
        elif diff_in:
            in_vec = abs(data[in_name+'p'] - data[in_name+'n'])
            in_name += 'p'
        else:
            in_vec = data[in_name]

        if data_out.any():
            out_vec = data_out
        elif diff_out:
            out_vec = abs(data[out_name+'p'] - data[out_name+'n'])
            out_name += 'p'
        else:
            out_vec = data[out_name]

        in_0, in_1 = self.get_pin_supply_values(in_name, data)
        out_0, out_1 = self.get_pin_supply_values(out_name, data)

        # evaluate t_start/t_stop
        if isinstance(t_start, str) or isinstance(t_stop, str):
            calc = self.get_calculator(data)
            if isinstance(t_start, str):
                t_start = calc.eval(t_start)
            if isinstance(t_stop, str):
                t_stop = calc.eval(t_stop)

        vth_in = (in_1 - in_0) * thres_delay + in_0
        vth_out = (out_1 - out_0) * thres_delay + out_0
        in_c = get_first_crossings(tvec, in_vec, vth_in, etype=in_edge, start=t_start, stop=t_stop,
                                   rtol=rtol, atol=atol)
        out_c = get_first_crossings(tvec, out_vec, vth_out, etype=out_edge, start=t_start,
                                    stop=t_stop, rtol=rtol, atol=atol)
        out_c -= in_c
        return out_c

    def calc_trf(self, data: SimData, out_name: str, out_rise: bool, allow_inf: bool = False,
                 t_start: Union[np.ndarray, float, str] = 0,
                 t_stop: Union[np.ndarray, float, str] = float('inf'),
                 data_in: np.ndarray = np.array([]), data_out: np.ndarray = np.array([])) -> np.ndarray:
        specs = self.specs
        logger = self.logger
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        out_0, out_1 = self.get_pin_supply_values(out_name, data)
        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        yvec = data[out_name]

        # evaluate t_start/t_stop
        if isinstance(t_start, str) or isinstance(t_stop, str):
            calc = self.get_calculator(data)
            if isinstance(t_start, str):
                t_start = calc.eval(t_start)
            if isinstance(t_stop, str):
                t_stop = calc.eval(t_stop)

        vdiff = out_1 - out_0
        vth_0 = out_0 + self._thres_lo * vdiff
        vth_1 = out_0 + self._thres_hi * vdiff
        if out_rise:
            edge = EdgeType.RISE
            t0 = get_first_crossings(tvec, yvec, vth_0, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)
            t1 = get_first_crossings(tvec, yvec, vth_1, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)
        else:
            edge = EdgeType.FALL
            t0 = get_first_crossings(tvec, yvec, vth_1, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)
            t1 = get_first_crossings(tvec, yvec, vth_0, etype=edge, start=t_start, stop=t_stop,
                                     rtol=rtol, atol=atol)

        has_nan = np.isnan(t0).any() or np.isnan(t1).any()
        has_inf = np.isinf(t0).any() or np.isinf(t1).any()
        if has_nan or (has_inf and not allow_inf):
            logger.warn(f'Got invalid value(s) in computing {edge.name} time of pin {out_name}.\n'
                        f't0:\n{t0}\nt1:\n{t1}')
            t1.fill(np.inf)
        else:
            t1 -= t0

        return t1

    def get_pin_supply_values(self, pin_name: str, data: SimData) -> Tuple[np.ndarray, np.ndarray]:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        gnd_pin, pwr_pin = self.get_pin_supplies(pin_name, pwr_domain)
        gnd_var = self.sup_var_name(gnd_pin)
        pwr_var = self.sup_var_name(pwr_pin)

        return self.get_param_value(gnd_var, data), self.get_param_value(pwr_var, data)

    def _bin_str_to_net(self, basename: str, bus_range: BusRange, bin_str: str,
                        pwr_domain: Mapping[str, Tuple[str, str]], src_pins: Set[str]) -> str:
        last_pin = ''
        cur_cnt = 0
        net_list = []
        for bus_idx, char in zip(bus_range, bin_str):
            cur_pin = f'{basename}<{bus_idx}>'
            if cur_pin not in src_pins:
                cur_pin = self.get_pin_supplies(cur_pin, pwr_domain)[int(char == '1')]

            if cur_pin == last_pin:
                cur_cnt += 1
            else:
                if last_pin:
                    net_list.append(last_pin if cur_cnt == 1 else f'<*{cur_cnt}>{last_pin}')
                last_pin = cur_pin
                cur_cnt = 1

        if last_pin:
            net_list.append(last_pin if cur_cnt == 1 else f'<*{cur_cnt}>{last_pin}')
        return ','.join(net_list)
