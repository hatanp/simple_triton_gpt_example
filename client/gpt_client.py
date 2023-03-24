# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os, sys
import numpy as np
import json
from tritonclient import utils as client_utils

import tritonclient.http as httpclient

import argparse

def prepare_input(name, input_dat):
    infer_input = httpclient.InferInput(name, input_dat.shape, client_utils.np_to_triton_dtype(input_dat.dtype))
    infer_input.set_data_from_numpy(input_dat)
    return infer_input



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",
                        type=str,
                        required=False,
                        default="localhost:8000",
                        help="Inference server URL. Default is localhost:8001.")
    parser.add_argument('-v',
                        "--verbose",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    args = parser.parse_args()

    try:
        triton_client = httpclient.InferenceServerClient(
            url=args.url, verbose=args.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)


    #Required model selected:
    model_name = "turkunlp_gpt3_small" #Could also be turkunlp_gpt3_13b but that is not in production API yet

    #Required inputs, comments in the form of (min:default:max):
    #Text
    input_0 = np.array([["Suomen paras kaupunki on"]]).astype(object)#No limitations, free text input with max length hard to define.
    #Float:
    temperature = np.array([[1.0]]).astype(np.float32)#0:1.0:2.0
    top_p = np.array([[1.0]]).astype(np.float32)#0:1.0:2.0
    #integer:
    top_k = np.array([[50]]).astype(np.int32)#0:50:200
    min_new_tokens = np.array([[1]]).astype(np.int32)#1:10:512
    max_new_tokens = np.array([[10]]).astype(np.int32)#1:50:512
    min_length = np.array([[1]]).astype(np.int32)#1:10:512
    max_length = np.array([[150]]).astype(np.int32)#1:150:512
    #Boolean
    do_sample = np.array([[True]]).astype(bool)#False:True:True

    inputs = []
    inputs = [
        prepare_input("INPUT_0",input_0),
        prepare_input("temperature",temperature),
        prepare_input("top_p",top_p),
        prepare_input("top_k",top_k),
        prepare_input("min_new_tokens",min_new_tokens),
        prepare_input("max_new_tokens",max_new_tokens),
        prepare_input("min_length",min_length),
        prepare_input("max_length",max_length),
        prepare_input("do_sample",do_sample),
    ]
    results = triton_client.infer(model_name=model_name,model_version="1",
                                  inputs=inputs)

    output_name = "OUTPUT_0"
    output0_data = [item.decode("utf-8") for item in results.as_numpy(output_name)]
    print(output0_data[0])
