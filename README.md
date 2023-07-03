# LoadShow
Artifacts for "LoadShow: Application Fingerprinting Based on the Combined Characteristics of CPU and GPU".

## Demo Videos
We evaluate LoadShow in the real world, demonstrating its ability to remotely recognize applications and their in-app actions.

For most applications, the total time required for LoadShow to collect a set of timing data and output the recognition results is about 12 to 30 seconds, depending on the parameter settings.

Link to the demo videos: https://drive.google.com/drive/folders/1i0h3epO5S6i4L1CvpZHhphLg8E0xbnkJ?usp=sharing

## dataset
9600k-2060: SOR33, SIR10, SOR8, SOR18, STR18-1, OSOR9

9600k-2060-multi_label: TOR12

e5-k2000: STR18-2

10700-550X: STR18-3

XXX: STR-Mac

XXX: STR-Pixel

XXX: HSCR18-i

## demo_sites
Fingerprinting functions for extracting the timing data.

## utils
Some files for data processing.

## Note
The fp_for_gpu.html in demo_sites is modified on the basis of https://github.com/drawnapart/drawnapart.
