Dear Reviewers,

**Please see the attached one-page PDF with added experimental results.**

We sincerely thank all the reviewers for their positive and constructive comments:  

* All reviewers appreciate that our paper introduces a simple yet effective training-free approach to disentangle textual and visual factors of CLIP image embedding in latent space (reviewer 1,2,3,4),  

* The observation of difference in mirror effects between visual objects and text on CLIP representation is novel (reviewer 1,3),  

* The experiment is sufficient and the results are comprehensive (reviewer 1,2,3). 

They also voiced several valid concerns. We have been diligently enhancing the paper on multiple fronts, addressing your concerns, the point-to-point response is provided. We summarize the changes updated below.

**1. Exploring the potential applications of MirrorCLIP in various downstream tasks.**

To explore MirrorCLIP's applications and downstream tasks, we combined it with RegionCLIP and SAM, for detection and text region segmentation. Specific examples can be found in Figure &#8544; in attached PDF. By using MirrorCLIP to get the disentangled visual region features of RegionCLIP, we can reduce the influence of textual factors and get more accurate detection results. By using the textual features obtained from MirrorCLIP to generate prompts for SAM, we can achieve text localization within images and perform preliminary text segmentation. These examples demonstrate the potential of MirrorCLIP for various downstream tasks.

**2. Further tested MirrorCLIP's disentanglement capability in various extreme scenarios.**

We further tested disentanglement capability of MirrorCLIP in three special scenarios, including typography with original and mirrored text, ordinary palindromes and special palindromes. We constructed corresponding datasets and conducted experiments, detailed results are shown in attached PDF.

According to the results, when handling ordinary palindromes, where the shape of the words changes before and after flipping ("did" to "bib"), MirrorCLIP can still achieve disentanglement performance comparable to that of handling other normal words. However, when handling special palindromes, where the shape of the words remains basically unchanged before and after flipping ("mom" to "mom"), MirrorCLIP struggles to achieve disentanglement. Yet, due to special palindromes are quite rare compared to other words, their impact is limited.

When handling typography with original and mirrored text, MirrorCLIP can still achieve disentanglement, but there is a noticeable decline in performance. Yet, compared to ordinary typography, typography with original and mirrored text is more like a targeted strong attack for our method, and not common in the real world. MirrorCLIP is primarily proposed as a disentanglement method, instead of a defense method.

We will discuss MirrorCLIP's performance in these scenarios in the Ablation and Limitation sections of the revision.

**3. More revisions that help enhance the clarity of the paper.**

* We will further clarify the differences between MirrorCLIP and other disentanglement methods.
* We will add the preliminary of CLIP, along with a more straightforward presentation of visual and textual components of images.
* We will correct and clarify all symbols and definitions that could lead to misunderstandings.

Please see our reviewer-specific feedback for detailed information.



## To reviewer 1
**W1:** It would be beneficial to delve deeper into the differences by comparing this approach to the existing CLIP-based method for textual and visual disentanglement in related works. 

**A1:** Thanks for your constructive advice. Compared to other CLIP-based works, our MirrorCLIP is the only training-free method without additional parameters and data, yet exhibits superior disentanglement performance. Moreover, MirrorCLIP remains the performance unaffected on the original datasets while others may degrade the performance.

Due to space constraints, we briefly introduced CLIP-based methods in L90. We will highlight the differences between MirrorCLIP and others in the revision. Specifically, Lemesle et al. introduced methodological tools from the cognitive science literature to assess the language biases of CLIP, and found that the textual and visual factors of an image do not share semantic representations in CLIP by presenting words that distort image classification across different categories levels. However, they cannot achieve disentangled representations of CLIP. Materzynska et al. disentangled visual and textual features by training different projection matrices and applying them to the CLIP outputs. However, it requires the introduction of additional model parameters and data for training; this also results in a performance decrease on the original datasets.

**W2:** Providing a preliminary introduction to CLIP would be better. Moreover, adding an image that introduces the concept of textual and visual objects of images will improve the clarity of the paper.

**A2:** Thanks for your advice. In our final version, we will add the preliminary of CLIP, along with a more straightforward presentation of visual and textual components of images to enhance the clarity of our work.

## To reviewer 2
**W1&W2&Q1:** It would be better to have a discussion on whether MirrorCLIP can be explored for other downstream tasks or applications.

**A1:** To explore MirrorCLIP's applications and downstream tasks, we combined it with RegionCLIP and SAM, for detection and text region segmentation. Specific examples can be found in Figure &#8544; in attached PDF.

For detection, RegionCLIP extends CLIP to learn region-level visual representations, allowing for detailed alignment between image regions and textual concepts. This capability supports region-based reasoning tasks, such as zero-shot and open-vocabulary object detection. However, the vanilla RegionCLIP is susceptible to textual components during object detection tasks. By using the MirrorCLIP framework to disentangle the region features of RegionCLIP, we can similarly reduce the influence of textual factors.  In Figure &#8544;(a), vanilla RegionCLIP mistakenly identified a price tag with text "papaya" as papaya. Moreover, after adding the text "television" on the laptop screen, vanilla RegionCLIP was misled and identified the laptop monitor as a television set. These errors were corrected by replacing the region features with the disentangled visual features obtained through MirrorCLIP. This highlights the potential of MirrorCLIP for applications in object detection.

For text region segmentation, by using the disentangled textual features obtained from MirrorCLIP to generate prompts for SAM, we can achieve text localization within images and perform preliminary text segmentation. Specific examples can be seen in Figure &#8544;(b). This shows that disentangled features through MirrorCLIP can be used for downstream tasks such as image segmentation. Our future work will continue to explore the applications of MirrorCLIP in various tasks.

## To reviewer 3
**W1:** Detailed experimental results of the disentangling framework when dealing with images containing flipped text were not provided in Ablation Experiment.

**A1:** Thanks for your thorough review of the paper. We show the detailed experimental results with images containing flipped text in Table &#8544; shown below. Based on the results of Table 6 and Table &#8544;, we can see that the vision features obtained through MirrorCLIP achieve high accuracy in image classification tasks when handling both normal text or flipped text. The results will be added in the revision.

**Table &#8544;:** Results of different features on image recognition with flipped text.

|       |original|typographic|
|  :---:  |  :---:  |  :---------------:  |
|   image features   |   61.38   | **55.97**|
|    flipped image features   |   61.59   | 37.56|
|   visual features  |**61.84**|50.30|
|    |    |    |

**W2:** More description of the potential applications of this disentanglement framework in practical tasks should be provided in conclusions.

**A2:** Thanks for your advice. We have initially explored object detection and text segmentation by combining MirrorCLIP with RegionCLIP and SAM. The results show the potential of MirrorCLIP for different downstream tasks or applications. Relevant examples are shown in Figure &#8544; in attached PDF. By using MirrorCLIP to get the disentangled visual region features of RegionCLIP, we can reduce the influence of textual factors and get more accurate detection results. By using the textual features obtained from MirrorCLIP to generate prompts for SAM, we can achieve text localization within images and perform preliminary text segmentation. In our revision, we will include a description of the potential applications of MirrorCLIP.

**W3:** Cross multiplication ($\times$) is used throughout equations in the manuscript, which may cause misunderstanding.

**A3:** Thank you for pointing out the notation issue. We will correct it and thoroughly check all mathematical notations in the revision.

## To reviewer 4

**W1&Q1:** Does … still work … text and its mirrored version? The proposed … might be easily circumvented.

**A1:** Yes, the disentangling mask still works. Although there is a 10.22 drop (59.71 to 49.49) in performance compared to the accuracy with ordinary typography (See Table &#8545; in attached PDF), MirrorCLIP still achieves disentanglement and defends against typographic attacks.

As shown in Figure &#8545;(b), we constructed a dataset that contains the original and the mirrored text. Our results revealed that, after adding original and mirrored text, the cosine similarity between image features before and after flipping also exhibited a great decrease from 0.9855 to 0.8566, as shown in Table &#8547;. As the core idea of our method is to leverage the lack of feature invariance in CLIP when flipping images, MirrorCLIP still can locate the textual components by comparing image features before and after flipping, as shown in the activation map in Figure &#8545;(b). Moreover, according to Table &#8545;, MirrorCLIP still achieves disentanglement with 9.73 improvements (39.76 to 49.49) compared to the baseline, and defends against typographic attacks. We suspect that besides semantic information, the positional information of text may also have some impact on the disentanglement of MirrorCLIP. Yet, the performance experiences a decline compared to the accuracy with ordinary typography, due to significant interference from the original and mirrored text.

We somehow disagree. Because defense/circumvention is not our main focus, and our MirrorCLIP is primarily proposed as a disentanglement method. Compared to ordinary typography, typography with original and mirrored text is a targeted strong attack for our method, and this is not common in the real world. We sincerely appreciate your thorough insights, will add a discussion of this in the limitation section, and explore defense methods against such a strong attack in the future.

**W2:** The hypothesis … overly generalized … palindromic words like "mom" … handwritten text …

**A2:** MirrorCLIP is capable of managing ordinary palindromes like "did" and "radar" or handwritten text, which change upon mirroring. However, it struggles to achieve disentanglement when dealing with special palindromes like "mom" and "wow". Yet, note that those special palindromes are extremely rare and hence basically have no impact on our hypothesis.

For the case of handwritten text, we have already conducted experiments on 3 real-world typographic datasets where the text is all handwritten and show excellent disentanglement results (Table 4 and Table 5).

For the case of palindromes, we categorized them into two types: ordinary palindromes, where the shape of the words changes before and after flipping ("did" to "bib"), and special palindromes, where the shape of the words remains basically unchanged ("mom" to "mom"). We constructed corresponding datasets: the ordinary palindrome dataset includes 26 words ("dad",  "madam", "radar", etc.), while the special palindrome dataset includes 5 words ("wow", "noon", "mom", "nun", "minim"). Both palindromes are illustrated in Figure &#8545;(c) and Figure &#8545;(d) in attached PDF. The results are shown in Table &#8546; in attached PDF. For ordinary palindromes, MirrorCLIP achieves disentanglement with 13.85 improvements compared to the baseline. This is a comparable improvement like other words. However, for special palindromes, MirrorCLIP struggles to achieve disentanglement and only improves the accuracy by 5.29. As special palindromes are quite rare compared to other words, according to the results in Table &#8546;, their impact is limited.

Thanks for pointing out this. We will include a description of the special palindrome scenario in the limitation section.

**W3-1&Q2:** Why … not produce … semantics of the typography?

**A3:** This issue is likely due to the limitations of the Stable UnCLIP model we used for feature visualization. It does not possess the capability to directly generate semantically relevant images when dealing with text-only images. The generated images are often meaningless characters, more examples are shown in Figure &#8545;(a) in attached PDF.

As seen in Figure 7 first row, after disentanglement, images generated with visual features do not carry textual components, and images generated with textual features do not carry visual components. This shows the effective disentanglement of MirrorCLIP.

**W3-2:** Whether … disentangles the semantics … text-form visual features.

**A4:** Our method can disentangle the textual semantics. This is verified through text recognition. According to the results in Tables 5 and 6, with disentangled textual features, the accuracy of text recognition improved significantly. This indicates the excellent disentanglement capability of MirrorCLIP for features with textual semantics, not only text-form visual features.

**Q3:** In Table 6, why … 61.03 when … zeroed out? Does … disentangling … insufficient?

**A5:** There might be some misunderstanding. **Text recognition accuracy when textual features are zeroed out is actually 23.18 (as shown after visual features (zero) in Table 6), not 61.03**, 61.03 is actually the text recognition accuracy when visual features are zeroed out. We would like to clarify that the label (zero) denotes textual features or visual features obtained by performing Hadamard product of textual or visual masks with image features, as defined in L247. We will clarify the meaning of the label (zero) more explicitly in the revision to avoid any confusion.

The disentanglement of textual features is sufficient based on the large decrease (from 72.51 to 5.29) in text recognition accuracy in Table 6. The text recognition accuracy of textual features obtained with the textual filter is 72.51 while the text recognition accuracy of visual features is 5.29. The large decrease in text recognition accuracy is due to the efficient removal of textual information.