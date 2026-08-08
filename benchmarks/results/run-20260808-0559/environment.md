# Environment — run-20260808-0559

## uname -a
```
Linux ubuntu-gpu-mi300x1-192gb-devcloud-atl1 7.0.0-27-generic #27-Ubuntu SMP PREEMPT_DYNAMIC Thu Jun 18 19:13:49 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
```

## rocm-smi
```
bash: line 15: rocm-smi: command not found
```

## rocm-smi --showtopo
```
bash: line 20: rocm-smi: command not found
```

## rocminfo
```
bash: line 25: rocminfo: command not found
```

## rocminfo | grep -i gfx
```
```

## amd-smi
```
bash: line 35: amd-smi: command not found
bash: line 35: amd-smi: command not found
```

## python / torch / hip
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch; print(torch.__version__, torch.version.hip)
    ^^^^^^^^^^^^
ModuleNotFoundError: No module named 'torch'
```

## nvidia-smi (expected absent on AMD box)
```
bash: line 45: nvidia-smi: command not found
```

## lscpu
```
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 57 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  20
On-line CPU(s) list:                     0-19
Vendor ID:                               GenuineIntel
Model name:                              INTEL(R) XEON(R) PLATINUM 8568Y+
CPU family:                              6
Model:                                   207
Thread(s) per core:                      1
Core(s) per socket:                      20
Socket(s):                               1
Stepping:                                2
BogoMIPS:                                4600.00
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm arch_perfmon pebs bts rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq dtes64 vmx ssse3 fma cx16 pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx_vnni avx512_bf16 wbnoinvd arat vnmi avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b fsrm md_clear serialize tsxldtrk avx512_fp16 arch_capabilities
Virtualization:                          VT-x
Hypervisor vendor:                       KVM
Virtualization type:                     full
L1d cache:                               640 KiB (20 instances)
L1i cache:                               640 KiB (20 instances)
L2 cache:                                80 MiB (20 instances)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-19
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Mitigation; Aligned branch/return thunks
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI SW loop, KVM SW loop
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Mitigation; TSX disabled
Vulnerability Vmscape:                   Not affected
```

## free -h
```
               total        used        free      shared  buff/cache   available
Mem:           235Gi       2.3Gi       234Gi       4.3Mi       818Mi       233Gi
Swap:             0B          0B          0B
```

## df -h
```
Filesystem      Size  Used Avail Use% Mounted on
tmpfs            48G  1.3M   48G   1% /run
/dev/vda1       698G  2.2G  696G   1% /
tmpfs           118G     0  118G   0% /dev/shm
tmpfs           118G     0  118G   0% /tmp
none            1.0M     0  1.0M   0% /run/credentials/systemd-journald.service
none            1.0M     0  1.0M   0% /run/credentials/systemd-resolved.service
/dev/vda13      989M  106M  816M  12% /boot
/dev/vda15      105M  6.3M   99M   7% /boot/efi
none            1.0M     0  1.0M   0% /run/credentials/systemd-networkd.service
none            1.0M     0  1.0M   0% /run/credentials/serial-getty@ttyS0.service
none            1.0M     0  1.0M   0% /run/credentials/getty@tty1.service
tmpfs            24G  8.0K   24G   1% /run/user/0
```

## OS release
```
PRETTY_NAME="Ubuntu 26.04 LTS"
NAME="Ubuntu"
VERSION_ID="26.04"
VERSION="26.04 LTS (Resolute Raccoon)"
VERSION_CODENAME=resolute
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=resolute
LOGO=ubuntu-logo
```
## lspci (GPU check)
```
00:01.0 VGA compatible controller: Red Hat, Inc. Virtio 1.0 GPU (rev 01)
83:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300X VF]
```

## python3 --version / pip
```
Python 3.14.4
bash: line 13: pip3: command not found
```
