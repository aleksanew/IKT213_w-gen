Image Matching Report (ORB_BF and SIFT_FLANN)

Threshold Used: 20

Comparing Efficiency
Time

ORB_BF is consistently faster across both datasets.
For most pairs, it completes in 17 ms, while SIFT_FLANN averages around 64 ms per fingerprint pair.
For the UiA images, ORB_BF was also faster with 109 ms vs 249 ms for SIFT.

Accuracy

SIFT_FLANN generally finds more good matches, but not all are correct, resulting in more false positives.
ORB_BF finds fewer matches but tends to be more precise.
For fingerprint verification, fewer false positives and negatives are better, meaning ORB_BF performs better overall in practical use.

Resources

SIFT descriptors are 128-dimensional floats, while ORB uses compact 32-byte binary descriptors.
This makes ORB_BF much more lightweight in memory and faster to match, while SIFT_FLANN is heavier but more robust.

ORB_BF Approach

Data_Check Accuracy: 85.00% (17/20)

Average Total Time: 0.017 s 

Average Descriptor Memory: 0.19 MB

Confusion Matrix (Data_Check)
<img width="900" height="750" alt="confmat_orb" src="https://github.com/user-attachments/assets/91835b9e-e604-4841-9703-77470f63fb75" />

Match Counts Plot (Data_Check)
<img width="1800" height="600" alt="match_counts_orb" src="https://github.com/user-attachments/assets/26d4d482-fa33-4e1f-b33f-cf57907abec6" />

Analysis:
ORB_BF shows clear separation between same and different fingerprints.
“Same” pairs have high match counts, while “different” pairs remain below threshold.
A few ambiguous samples exist close to the decision line.

SIFT_FLANN Approach

Data_Check Accuracy: 95.00% (19/20)

Average Total Time: 0.064 s

Average Descriptor Memory: 1.85 MB

Confusion Matrix (Data_Check)
<img width="900" height="750" alt="confmat_sift" src="https://github.com/user-attachments/assets/06ff28e5-8edb-4965-8cd4-742e12328f78" />

Match Counts Plot (Data_Check)
<img width="1800" height="600" alt="match_counts_sift" src="https://github.com/user-attachments/assets/cc752360-405e-47d0-95d6-c59eda833c6b" />

Analysis:
SIFT_FLANN finds more matches overall, often hundreds on same fingerprints.
However, it also detects more “good matches” on different fingerprints, which may not be truly correct.
This increases robustness but reduces reliability when a strict threshold is used.

UiA Matching
ORB_BF Approach

Predicted: No Match

Good Matches: 9

Keypoints: 4000 × 4000

Descriptor Memory: 0.24 MB

Time: 0.109 s

Observation:
ORB_BF detects few reliable matches between the two UiA photos.
This makes sense due to large differences in viewpoint, brightness, and season.

SIFT_FLANN Approach

Predicted: No Match

Good Matches: 21

Keypoints: 4000 × 4000

Descriptor Memory: 3.91 MB

Time: 0.249 s

Observation:
SIFT_FLANN finds more “good” correspondences.
This fits the overall pattern.
