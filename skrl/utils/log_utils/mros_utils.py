import mros
from mros.std_msgs.msg import Float32MultiArray


class MrosUtils:
    def __init__(self, mros_node_name: str):
        self.mros_node_name = mros_node_name
        self.mros_debug_pub_dict = dict()
        self.mros_debug_array_dict = dict()
        # self.action_without_head_idx_order = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
        # self.joint_idx_order = [0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11, 16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30]

        mros.init(self.mros_node_name)

    def get_debug_topic_list(self):
        return list(self.mros_debug_pub_dict.keys())

    def register_debug_pub(self, topic_name: str, msg_type: type = Float32MultiArray):
        self.mros_debug_pub_dict[topic_name] = mros.advertise(
            f"/debug_info/{self.mros_node_name}/{topic_name}", msg_type
        )
        self.mros_debug_array_dict[topic_name] = msg_type()

    def publish_debug(self, topic_name: str, data: list):
        if topic_name not in self.mros_debug_pub_dict:  
            raise ValueError(f"Topic {topic_name} not registered")
        self.mros_debug_array_dict[topic_name].data = data
        self.mros_debug_pub_dict[topic_name].publish(
            self.mros_debug_array_dict[topic_name]
        )
    
    # def publish_rearanged_debug(self, topic_name: str, data: list):
    #     if topic_name not in self.mros_debug_pub_dict:
    #         raise ValueError(f"Topic {topic_name} not registered")
    #     self.mros_debug_array_dict[topic_name].data = [data[i] for i in self.joint_idx_order] 
    #     self.mros_debug_pub_dict[topic_name].publish(
    #         self.mros_debug_array_dict[topic_name]
    #     )

    # def publish_rearanged_debug_without_head(self, topic_name: str, data: list):
    #     if topic_name not in self.mros_debug_pub_dict:
    #         raise ValueError(f"Topic {topic_name} not registered")
    #     self.mros_debug_array_dict[topic_name].data = [data[i] for i in self.action_without_head_idx_order] 
    #     self.mros_debug_pub_dict[topic_name].publish(
    #         self.mros_debug_array_dict[topic_name]
    #     )

