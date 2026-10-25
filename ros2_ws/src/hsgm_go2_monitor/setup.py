from setuptools import setup
setup(
    name="hsgm_go2_monitor", version="0.1.0", packages=["hsgm_go2_monitor"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/hsgm_go2_monitor"]),
        ("share/hsgm_go2_monitor", ["package.xml"]),
    ],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="HSGM maintainers", maintainer_email="codex@users.noreply.github.com",
    description="Subscriber-only Go2 telemetry", license="Apache-2.0",
    entry_points={"console_scripts": ["monitor = hsgm_go2_monitor.monitor:main"]},
)
