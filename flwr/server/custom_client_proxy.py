# from typing import Dict, Optional, Any
from typing import Dict, List, Optional, Tuple, Union, Any
from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Properties,
    ReconnectIns,
    Properties as Props,
)

from flwr.server.client_proxy import ClientProxy

class CustomClientProxy(ClientProxy):
    def __init__(self, cid: str):
        super().__init__(cid)
        self.messages: Dict[str, str] = {}
        self.parameters: Dict[str, Any] = {}  # Simulated model parameters
        self.properties: Props = {}  # Simulated properties

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> GetPropertiesRes:
        # Return simulated properties
        return GetPropertiesRes(properties=self.properties)

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> GetParametersRes:
        # Return simulated model parameters
        return GetParametersRes(parameters=self.parameters)

    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> FitRes:
        # Simulate fitting process by updating model parameters
        self.parameters.update(ins.parameters)
        # Simulate successful fitting
        return FitRes(num_examples=len(ins.parameters.get('data', [])))

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> EvaluateRes:
        # Simulate evaluation and return a dummy loss and accuracy
        loss = 0.1
        accuracy = 0.9
        return EvaluateRes(loss=loss, accuracy=accuracy)

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> DisconnectRes:
        # Simulate a successful reconnect
        return DisconnectRes(status="reconnected")

    def notify(self, message: str) -> None:
        """
        Notify the client with a message.

        Args:
            message (str): The notification message to send to the client.
        """
        # In this example, we're simply storing the messages in a dictionary
        self.messages[self.cid] = message
        print(f"Notification sent to client {self.cid}: {message}")
        # Here you might use an actual messaging or notification system, such as gRPC, HTTP, etc.
