# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Tests for signal_to_noise_ratio.py."""

from absl.testing import absltest
import buffling.signal_to_noise_ratio as snr
import numpy as np


class SignalToNoiseRatioTest(absltest.TestCase):

  def test_generate_transition_matrix(self):
    P = snr.generate_transition_matrix(n_pub=1, p=0.1)
    # self.assertAlmostEqual(P[0,1], 0.1)
    np.testing.assert_array_almost_equal(
      P, np.array([[.9,.1],[.1,.9]]), 
      err_msg='Two VoC are not the same with the same random seed.')


if __name__ == '__main__':
  absltest.main()