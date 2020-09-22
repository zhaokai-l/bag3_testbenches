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

from typing import Any, Union, Sequence, Tuple, Optional, Mapping, Iterable, List, Set

# from itertools import chain

# import numpy as np

# from pybag.core import get_cdba_name_bits

# from bag.simulation.data import SimData, AnalysisType

from bag3_liberty.data import parse_cdba_name, BusRange

# from ..data.tran import EdgeType, get_first_crossings
from .base import TranTB


class AnalogTranTB(TranTB):
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
        pin_values: Mapping[str, int] = specs['pin_values']
        stimuli_list: Sequence[Mapping[str, Any]] = specs.get('stimuli_list', [])
        load_list: Sequence[Mapping[str, Any]] = specs.get('load_list', [])

        src_list = []
        src_pins = set()
        self.get_stimuli_sources(stimuli_list, src_list, src_pins)
        self.get_bias_sources(sup_values, src_list, src_pins)
        self.get_loads(load_list, src_list)

        dut_conns = self.get_dut_conns(dut_pins, src_pins, pin_values)
        return dict(
            dut_lib=sch_params.get('dut_lib', ''),
            dut_cell=sch_params.get('dut_cell', ''),
            dut_params=sch_params.get('dut_params', None),
            dut_conns=dut_conns,
            vbias_list=[],
            src_list=src_list,
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

            pwr_name, gnd_name = self.get_pin_supplies(pin, pwr_domain)

            src_type: str = wave_params['src_type']
            table: Mapping[str, Any] = wave_params['table']
            # for v, k in table:
            #     src_table.update(k=self.get_sim_param_string(v))
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
                      pin_values: Mapping[str, int]) -> Mapping[str, str]:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        ans = {}
        for pin_name in dut_pins:
            pin_val: Optional[int] = pin_values.get(pin_name, None)
            basename, bus_range = parse_cdba_name(pin_name)
            if bus_range is None:
                # scalar pins
                if pin_name in src_pins or pin_val is None:
                    ans[pin_name] = pin_name
                else:
                    ans[pin_name] = self.get_pin_supplies(pin_name, pwr_domain)[pin_val]
            else:
                # bus pins
                if pin_val is None:
                    # no bias values specified
                    ans[pin_name] = pin_name
                else:
                    nlen = len(bus_range)
                    bin_str = bin(pin_val)[2:].zfill(nlen)
                    ans[pin_name] = self._bin_str_to_net(basename, bus_range, bin_str, pwr_domain,
                                                         src_pins)

        return ans

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
