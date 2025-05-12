# 2025-05-12T14:49:00.400843
import vitis

client = vitis.create_client()
client.set_workspace(path="lab1")

comp = client.create_hls_component(name = "hls_conv2d",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

vitis.dispose()

