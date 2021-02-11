# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Type, Union, Dict, Optional, Any, cast, Sequence, Mapping
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimNetlistInfo, netlist_info_from_dict
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.design.module import Module
from bag.concurrent.util import GatherHelper

from ...schematic.cap_tb_ac import bag3_testbenches__cap_tb_ac


class CapACTB(TestbenchManager):
    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        return bag3_testbenches__cap_tb_ac

    def get_netlist_info(self) -> SimNetlistInfo:
        sweep_var: str = self.specs.get('sweep_var', 'freq')
        sweep_options: Mapping[str, Any] = self.specs['sweep_options']
        ac_options: Mapping[str, Any] = self.specs.get('ac_options', {})
        save_outputs: Sequence[str] = self.specs.get('save_outputs', ['plus', 'minus'])
        ac_dict = dict(type='AC',
                       param=sweep_var,
                       sweep=sweep_options,
                       options=ac_options,
                       save_outputs=save_outputs,
                       )

        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [ac_dict]
        return netlist_info_from_dict(sim_setup)


class CapACMeas(MeasurementManager):
    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo):
        raise NotImplementedError

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance):
        raise NotImplementedError

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]):
        raise NotImplementedError

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Mapping[str, Any]:
        helper = GatherHelper()
        for idx in range(2):
            helper.append(self.async_meas_case(name, sim_dir, sim_db, dut, idx))

        meas_results = await helper.gather_err()
        ans = self.compute_passives(meas_results)

        return ans

    @staticmethod
    def compute_passives(meas_results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        freq0 = meas_results[0]['freq']
        freq1 = meas_results[1]['freq']
        assert np.isclose(freq0, freq1).all()

        # vm0 = (zc * zpm) / (zc + zpp + zpm)
        # vp0 = - (zc * zpp) / (zc + zpp + zpm)
        vm0 = meas_results[0]['minus']
        vp0 = meas_results[0]['plus']

        # vm1 = - (zpp * zpm) / (zc + zpp + zpm)
        # vp1 = - ((zc + zpm) * zpp) / (zc + zpp + zpm)
        vm1 = meas_results[1]['minus']
        vp1 = meas_results[1]['plus']

        # --- Find zc, zpp, zpm using vm0, vp0, vm1 --- #
        # - vp0 / vm0 = zpp / zpm = const_a  ==> zpp = const_a * zpm
        const_a = - vp0 / vm0
        # vp0 / vm1 = zc / zpm = const_b  ==> zc = const_b * zpm
        const_b = vp0 / vm1

        # vp0 = - (const_b * const_a * zpm) / (const_b + const_a + 1)
        zpm = - vp0 * (const_b + const_a + 1) / (const_b * const_a)
        zpp = const_a * zpm
        zc = const_b * zpm

        # --- Verify vp1 is consistent --- #
        vp1_calc = - ((zc + zpm) * zpp) / (zc + zpp + zpm)
        if not np.isclose(vp1, vp1_calc, rtol=1e-3).all():
            plt.loglog(freq0, np.abs(vp1), label='measured')
            plt.loglog(freq0, np.abs(vp1_calc), 'g--', label='calculated')
            plt.xlabel('Frequency (in Hz)')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

        return dict(
            cc=estimate_cap(freq0, zc),
            cpp=estimate_cap(freq0, zpp),
            cpm=estimate_cap(freq0, zpm),
        )

    async def async_meas_case(self, name: str, sim_dir: Path, sim_db: SimulationDB, dut: Optional[DesignInstance],
                              case_idx: int) -> Dict[str, Any]:
        if case_idx == 0:
            sup_conns = [('PLUS', 'plus'), ('MINUS', 'minus')]
        elif case_idx == 1:
            sup_conns = [('PLUS', 'plus'), ('MINUS', 'VSS')]
        else:
            raise ValueError(f'Invalid case_idx={case_idx}')

        tbm_specs = dict(
            **self.specs['tbm_specs']['ac_meas'],
            sim_envs=self.specs['sim_envs'],
        )
        tbm = cast(CapACTB, self.make_tbm(CapACTB, tbm_specs))
        tbm_name = f'{name}_{case_idx}'
        tb_params = dict(
            extracted=self.specs['tbm_specs'].get('extracted', True),
            sup_conns=sup_conns,
        )
        sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir / tbm_name, dut, tbm,
                                                          tb_params=tb_params)
        data = sim_results.data
        return dict(freq=data['freq'], plus=np.squeeze(data['plus']), minus=np.squeeze(data['minus']))


def estimate_cap(freq: np.ndarray, zc: np.ndarray) -> float:
    fit = np.polyfit(2 * np.pi * freq, - 1 / np.imag(zc), 1)
    return fit[0]
