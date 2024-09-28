# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
# ==============================================================================
"""Flower ClientProxy implementation using gRPC bidirectional streaming."""


from typing import Optional, Any
import json
from flwr.common import FitIns, Parameters, Scalar

from flwr import common
from flwr.common import serde
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    ServerMessage,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import (
    GrpcBridge,
    InsWrapper,
    ResWrapper,
)


class GrpcClientProxy(ClientProxy):
    """Flower ClientProxy that uses gRPC to delegate tasks over the network."""

    def __init__(
        self,
        cid: str,
        bridge: GrpcBridge,
    ):
        super().__init__(cid)
        self.bridge = bridge

    def get_properties(
        self,
        ins: common.GetPropertiesIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.GetPropertiesRes:
        """Request client's set of internal properties."""
        get_properties_msg = serde.get_properties_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_properties_ins=get_properties_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_properties_res = serde.get_properties_res_from_proto(
            client_msg.get_properties_res
        )
        return get_properties_res

    def get_parameters(
        self,
        ins: common.GetParametersIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        get_parameters_msg = serde.get_parameters_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_parameters_ins=get_parameters_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_parameters_res = serde.get_parameters_res_from_proto(
            client_msg.get_parameters_res
        )
        return get_parameters_res

    def fit(
        self,
        ins: common.FitIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.FitRes:
        """Refine the provided parameters using the locally held dataset."""
        fit_ins_msg = serde.fit_ins_to_proto(ins)

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(fit_ins=fit_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        fit_res = serde.fit_res_from_proto(client_msg.fit_res)
        return fit_res

    def evaluate(
        self,
        ins: common.EvaluateIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        evaluate_msg = serde.evaluate_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(evaluate_ins=evaluate_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        evaluate_res = serde.evaluate_res_from_proto(client_msg.evaluate_res)
        return evaluate_res

    def reconnect(
        self,
        ins: common.ReconnectIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        reconnect_ins_msg = serde.reconnect_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(reconnect_ins=reconnect_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        disconnect = serde.disconnect_res_from_proto(client_msg.disconnect_res)
        return disconnect

    # def send_object(
    #         self,
    #         obj: Any,  # Object to send
    #         timeout: Optional[float] = None,
    #         group_id: Optional[int] = None,
    #     ) -> Any:
    #         """Send an arbitrary object to the client."""
            
    #         # 1. Serialize the object using JSON
    #         obj_json = json.dumps(obj)  # Convert the object to JSON string
            
    #         # 2. Create a ServerMessage containing the serialized object
    #         server_message = ServerMessage(custom_object=obj_json)  # Add custom field for object
            
    #         # 3. Wrap the message in InsWrapper to send it to the client
    #         res_wrapper: ResWrapper = self.bridge.request(
    #             ins_wrapper=InsWrapper(server_message=server_message, timeout=timeout)
    #         )
            
    #         # 4. Extract the response message from the client
    #         client_msg: ClientMessage = res_wrapper.client_message
            
    #         # 5. Deserialize the response message back to a Python object
    #         response_object = json.loads(client_msg.custom_object)  # Convert JSON string back to Python object
            
    #         return response_object
    def send_object(
        self,
        obj: Any,  # The object to send
        timeout: Optional[float] = None,
        group_id: Optional[int] = None,
    ):
        """Send custom data using the FitIns message."""
        
        # 1. Serialize the object to JSON
        obj_json = json.dumps(obj)

        # 2. Prepare a FitIns message (we can use the `config` field to send custom data)
        parameters = Parameters(
            tensors=[],  # Empty model parameters since we're sending custom data
            tensor_type= "numpy.ndarray"  # Provide a suitable tensor type (e.g., FLOAT32)
        )
        
        # 3. Prepare a FitIns message using the Parameters object and custom config
        fit_ins = FitIns(
            parameters=parameters,
            config={"custom_data": obj_json}
          )  #
        fit_ins_msg = serde.fit_ins_to_proto(fit_ins)
        # 3. Use the fit method to send the custom data to the client
        res_wrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(fit_ins=fit_ins_msg),
                timeout=timeout
            )
        )
        
        # 4. Process the response from the client (if necessary)
        # client_msg = res_wrapper.client_message

        # print("Response from client:", client_msg)

    # def send_object(self, obj: dict, timeout: Optional[float] = None, group_id: Optional[int] = None):
    #     """Send a custom object to the client."""
        
    #     # 1. Serialize the object to JSON
    #     obj_json = json.dumps(obj)

    #     # 2. Prepare the Parameters object (even if empty, it must be constructed correctly)
    #     parameters = Parameters(
    #         tensors=[],  # Sending an empty tensor list as we're not sending parameters
    #         tensor_type=float  # A required tensor type (you can use other types based on your need)
    #     )

    #     # 3. Prepare the FitIns message, with parameters and the object as part of the config
    #     fit_ins = FitIns(
    #         parameters=parameters,
    #         config={"custom_data": obj_json}  # Send the object as a JSON string in config
    #     )

    #     # 4. Create a ServerMessage with the FitIns message, using a dict if required by your gRPC structure
    #     server_message = ServerMessage(
    #         fit_ins=fit_ins  # Now properly initialized as part of the ServerMessage
    #     )

    #     # 5. Use the gRPC bridge to send the message
    #     res_wrapper = self.bridge.request(
    #         ins_wrapper=InsWrapper(
    #             server_message=server_message,
    #             timeout=timeout
    #         )
    #     )

    #     # 6. Process the client response
    #     client_msg = res_wrapper.client_message
    #     print("Response from client:", client_msg)
