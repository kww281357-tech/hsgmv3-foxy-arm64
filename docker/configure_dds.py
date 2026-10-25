"""Generate constrained CycloneDDS 0.10 configuration; never change host networking."""
import os
import re
import socket
import xml.etree.ElementTree as ET
from pathlib import Path


def build_config(interface, domain, available):
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", interface) or interface not in available:
        raise ValueError("GO2_INTERFACE must name an existing, verified robot-facing NIC")
    if not domain.isdecimal() or not 0 <= int(domain) <= 232:
        raise ValueError("ROS_DOMAIN_ID must be in 0..232")
    root = ET.Element("CycloneDDS")
    node = ET.SubElement(root, "Domain", id=domain)
    general = ET.SubElement(node, "General")
    interfaces = ET.SubElement(general, "Interfaces")
    ET.SubElement(interfaces, "NetworkInterface", name=interface)
    ET.SubElement(general, "AllowMulticast").text = "false" if interface == "lo" else "true"
    discovery = ET.SubElement(node, "Discovery")
    ET.SubElement(discovery, "ParticipantIndex").text = "auto"
    ET.SubElement(discovery, "MaxAutoParticipantIndex").text = "9"
    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":
    Path("/tmp/hsgm-cyclonedds.xml").write_text(build_config(
        os.environ.get("GO2_INTERFACE", "lo"), os.environ.get("ROS_DOMAIN_ID", "0"),
        {name for _, name in socket.if_nameindex()}))
