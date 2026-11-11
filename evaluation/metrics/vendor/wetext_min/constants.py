# Copyright (c) 2025 Zhendong Peng (pzd17@tsinghua.org.cn)
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

# Retain pinned token ordering constants only.
EOS = "<EOS>"
TN_ORDERS = {
    "date": ["year", "month", "day"],
    "fraction": ["denominator", "numerator"],
    "measure": ["denominator", "numerator", "value"],
    "money": ["value", "currency"],
    "time": ["noon", "hour", "minute", "second"],
}
EN_TN_ORDERS = {
    "date": ["preserve_order", "text", "day", "month", "year"],
    "money": ["integer_part", "fractional_part", "quantity", "currency_maj"],
}
ITN_ORDERS = {
    "date": ["year", "month", "day", "preserve_order"],
    "fraction": ["sign", "numerator", "denominator"],
    "measure": ["numerator", "denominator", "value", "units"],
    "money": ["currency", "value", "decimal", "quantity"],
    "time": ["hour", "minute", "second", "noon", "zone"],
    "telephone": ["country_code", "number_part"],
    "electronic": ["username", "domain", "protocol"],
}
EN_ITN_ORDERS = ITN_ORDERS
