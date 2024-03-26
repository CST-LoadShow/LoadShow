# LoadShow
Artifacts for "LoadShow: Real-World Fine-Grained Application Fingerprinting via Multi-Source Load Pattern Extraction".

## Demo Videos
We evaluate LoadShow in the real world, demonstrating its ability to remotely recognize applications and their in-app actions.

For most applications, the total time required for LoadShow to collect a set of timing data and output the recognition results is about 12 to 30 seconds, depending on the parameter settings.

Link to the demo videos: https://drive.google.com/drive/folders/1i0h3epO5S6i4L1CvpZHhphLg8E0xbnkJ?usp=sharing

## Android App implementation of LoadShow

Data: 26/3/2024   Folder: android_version

We have recently implemented the Android App version of LoadShow, and the timer resolution has been improved from `ms` level in the web implementation to `ns` level in the Android App implementation.

Under the same experimental setup (same Android phone, same 20 test Apps), the recognition accuracy improved from 85.71% (web version) to 95.09% (Android App version).
We will upload the relevant code and dataset for this part once it is organized.

## Datasets
9600k-2060: SOR33, SOR8, SOR18, STR18-1, OSOR9

9600k-2060-behavior: SIR10

9600k-2060-multi_label: TOR12

e5-k2000: STR18-2

10700-550X: STR18-3

Mac1: STR-Mac1, Mac2: STR-Mac2

Pixel: STR-Pixel

410-X: HSCR18-i

## demo_sites
Projects for real-world evaluation and webpages that contain fingerprinting functions used for data extraction.

## utils
Files for data processing.
