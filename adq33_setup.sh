sudo apt update
# DKMS + build tools
sudo apt install -y dkms build-essential

# Jetson kernel headers (works for most L4T/JetPack):
sudo apt install -y nvidia-l4t-kernel-headers

# (Optional) If the above isn’t available or you’re on a custom kernel:
# sudo apt install -y linux-headers-$(uname -r) || true
dkms --version         # should print a version (>= 1.95 is fine)
ls /usr/src | grep -E 'tegra|linux-headers'

sudo dpkg -i spd-adq-pci-dkms_1.26_all.deb
sudo dpkg -i libadq0_2025.1.275.gaacf6005_arm64.deb
sudo dpkg -i adqtools_2025.1.275.gaacf60053_arm64.deb
python3 -m pip install ./pyadq-2025.1-py3-none-any.whl

sudo usermod -a -G adq $USER
sudo reboot


