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
import tritongrpcclient
#from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient import utils as client_utils

import tritonclient.http as httpclient

import argparse


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.
    
    """
    return np.fromfile(img_path, dtype='uint8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=False,
                        default="turkunlp_gpt3",
                        help="Model name")
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
        #triton_client = tritonclient.grpc.InferenceServerClient(
        #    url=args.url, verbose=args.verbose)
        triton_client = httpclient.InferenceServerClient(
            url=args.url, verbose=args.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    inputs = []
    outputs = []
    input_name = "INPUT_0"
    output_name = "OUTPUT_0"
    
    input_0 = np.array([["Suomen paras kaupunki on"]]).astype(object)
    
    infer_input = httpclient.InferInput(input_name, input_0.shape, client_utils.np_to_triton_dtype(input_0.dtype))
    infer_input.set_data_from_numpy(input_0)
    
    inputs = [infer_input]

    results = triton_client.infer(model_name=args.model_name,
                                  inputs=inputs)

    print(results)
    output0_data = [item.decode("utf-8") for item in results.as_numpy(output_name)]
    print(output0_data[0])
    #maxs = np.argmax(output0_data, axis=1)
