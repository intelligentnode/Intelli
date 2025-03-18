from enum import Enum
from typing import Callable, Dict, List, Any, Union, Optional


class ConnectorMode(Enum):
    """Enum for different dynamic connector modes."""

    CONTENT_BASED = "content"
    LENGTH_BASED = "length"
    ERROR_BASED = "error"
    TYPE_BASED = "type"
    CUSTOM = "custom"


class DynamicConnector:
    """
    A class to handle dynamic routing in a flow based on the output of a task.
    Allows for up to 4 possible paths based on the decision function.
    """

    def __init__(
        self,
        decision_fn: Callable[[Any, str], str],
        destinations: Dict[str, str],
        name: str = "dynamic_connector",
        description: str = "Routes based on previous output",
        mode: ConnectorMode = ConnectorMode.CUSTOM,
    ):
        """
        Initialize a dynamic connector.

        Args:
            decision_fn: A function that takes (output, output_type) and returns a destination key
            destinations: A dictionary mapping destination keys to task names (max 4)
            name: Name of the connector (for visualization)
            description: Description of the connector logic
            mode: The connector mode (for visualization)
        """
        if destinations and len(destinations) > 4:
            raise ValueError("Dynamic connector can have at most 4 destinations")

        self.decision_fn = decision_fn
        self.destinations = destinations
        self.name = name
        self.description = description
        self.mode = mode

    def get_next_task(self, output: Any, output_type: str) -> Optional[str]:
        """
        Determine the next task based on the output and its type.

        Args:
            output: The output from the previous task
            output_type: The type of the output (text, image, audio, etc.)

        Returns:
            The name of the next task to execute, or None if no matching destination
        """
        try:
            destination_key = self.decision_fn(output, output_type)
            return self.destinations.get(destination_key)
        except Exception as e:
            print(f"Error in dynamic connector {self.name}: {e}")
            return None
