from typing import Any, Dict, AnyStr


class Node:
    def __init__(self, value: Any, attributes: Dict[AnyStr, Any] = None) -> None:
        """
        Node of a graph

        :param value: the value of the node
        :param attributes: attributes of the node
        """
        self.degree = 0
        self.value = value

        self.attributes = (
            attributes.update({"degree": 0, "in_degree": 0, "out_degree": 0})
            if attributes is not None
            else {"degree": 0, "in_degree": 0, "out_degree": 0}
        )

    def get_value(self) -> Any:
        """
        get the value of the node

        :return: value of the node
        :rtype: Any
        """
        return self.value

    def get_attributes(self) -> Dict[AnyStr, Any]:
        """
        get the attributes of the node

        :return: attributes of the node
        :rtype: Any
        """
        return self.attributes

    def get_attribute(self, attribute: AnyStr) -> Any:
        """
        get one specific attribute by name

        :param attribute: the attribute's name
        :return: the attribute requested
        :rtype: Any
        """
        return self.attributes.get(attribute)

    def set_attribute(self, attribute: AnyStr, value: Any) -> None:
        """
        set one specific attribute by name

        :param attribute: the attribute to be set
        :param value: the value the attribute is set to
        :return: None
        """
        self.attributes.update({attribute: value})

    def get_degree(self) -> int:
        """
        get the degree of the node

        :return: degree of node
        :rtype: int
        """
        return self.degree

    def set_degree(self, degree: int) -> None:
        """
        set the degree of the node

        :param degree: degree of the node
        :return: None
        """

        self.degree = degree
        self.attributes["degree"] = degree

    def increase_degree(self, by: int) -> None:
        """
        increase the degree of the node

        :param by: increment the node's degree is increased by
        :return: None
        """

        self.degree += by
        self.attributes["degree"] = self.degree

    def __str__(self) -> str:
        return (
            f'Node({self.value}){", " if self.attributes else ""}'
            f'{", ".join([str((key, value)) for key, value in self.attributes.items()])}'
        )
