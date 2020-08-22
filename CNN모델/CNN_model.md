```python
import tensorflow as tf
from tensorflow import keras
```


```python
tf.__version__
```




    '2.1.0'




```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
```

    Using TensorFlow backend.
    


```python
!pip install pillow
!pip install sklearn
```

    Requirement already satisfied: pillow in c:\users\admin\anaconda3\envs\tf\lib\site-packages (7.1.1)
    Requirement already satisfied: sklearn in c:\users\admin\anaconda3\envs\tf\lib\site-packages (0.0)
    Requirement already satisfied: scikit-learn in c:\users\admin\anaconda3\envs\tf\lib\site-packages (from sklearn) (0.22.2.post1)
    Requirement already satisfied: joblib>=0.11 in c:\users\admin\anaconda3\envs\tf\lib\site-packages (from scikit-learn->sklearn) (0.14.1)
    Requirement already satisfied: scipy>=0.17.0 in c:\users\admin\anaconda3\envs\tf\lib\site-packages (from scikit-learn->sklearn) (1.4.1)
    Requirement already satisfied: numpy>=1.11.0 in c:\users\admin\anaconda3\envs\tf\lib\site-packages (from scikit-learn->sklearn) (1.18.2)
    


```python
import PIL
from sklearn.model_selection import train_test_split
import random
import os
```


```python
print(os.listdir("D:/workspace/AOI/청북공장 이미지"))
```

    ['barcode', 'image', 'image.zip', 'ng', 'ok', 'ok - 복사본', 'OK_all', 'simulation', 'test', 'train', 'train_ver2', 'ver3', '원본파일']
    

## define Constants


```python
FAST_RUN = False
IMAGE_WIDTH = 236
IMAGE_HEIGHT = 236
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
```

## prepare training data


```python
filenames = os.listdir("D:/workspace/AOI/청북공장 이미지/train")
```


```python
categories = []
```


```python
filenames
```




    ['NG (1).jpg',
     'NG (10).jpg',
     'NG (100).jpg',
     'NG (101).jpg',
     'NG (102).jpg',
     'NG (103).jpg',
     'NG (104).jpg',
     'NG (105).jpg',
     'NG (106).jpg',
     'NG (107).jpg',
     'NG (108).jpg',
     'NG (109).jpg',
     'NG (11).jpg',
     'NG (110).jpg',
     'NG (111).jpg',
     'NG (112).jpg',
     'NG (113).jpg',
     'NG (114).jpg',
     'NG (115).jpg',
     'NG (116).jpg',
     'NG (117).jpg',
     'NG (118).jpg',
     'NG (119).jpg',
     'NG (12).jpg',
     'NG (120).jpg',
     'NG (121).jpg',
     'NG (122).jpg',
     'NG (123).jpg',
     'NG (124).jpg',
     'NG (125).jpg',
     'NG (126).jpg',
     'NG (127).jpg',
     'NG (128).jpg',
     'NG (129).jpg',
     'NG (13).jpg',
     'NG (130).jpg',
     'NG (131).jpg',
     'NG (132).jpg',
     'NG (133).jpg',
     'NG (134).jpg',
     'NG (135).jpg',
     'NG (136).jpg',
     'NG (137).jpg',
     'NG (138).jpg',
     'NG (139).jpg',
     'NG (14).jpg',
     'NG (140).jpg',
     'NG (141).jpg',
     'NG (142).jpg',
     'NG (143).jpg',
     'NG (144).jpg',
     'NG (145).jpg',
     'NG (146).jpg',
     'NG (147).jpg',
     'NG (148).jpg',
     'NG (149).jpg',
     'NG (15).jpg',
     'NG (150).jpg',
     'NG (151).jpg',
     'NG (152).jpg',
     'NG (153).jpg',
     'NG (154).jpg',
     'NG (155).jpg',
     'NG (156).jpg',
     'NG (157).jpg',
     'NG (158).jpg',
     'NG (159).jpg',
     'NG (16).jpg',
     'NG (160).jpg',
     'NG (161).jpg',
     'NG (162).jpg',
     'NG (163).jpg',
     'NG (164).jpg',
     'NG (165).jpg',
     'NG (166).jpg',
     'NG (167).jpg',
     'NG (168).jpg',
     'NG (169).jpg',
     'NG (17).jpg',
     'NG (170).jpg',
     'NG (171).jpg',
     'NG (172).jpg',
     'NG (173).jpg',
     'NG (174).jpg',
     'NG (175).jpg',
     'NG (176).jpg',
     'NG (177).jpg',
     'NG (178).jpg',
     'NG (179).jpg',
     'NG (18).jpg',
     'NG (180).jpg',
     'NG (181).jpg',
     'NG (182).jpg',
     'NG (183).jpg',
     'NG (184).jpg',
     'NG (185).jpg',
     'NG (186).jpg',
     'NG (187).jpg',
     'NG (188).jpg',
     'NG (189).jpg',
     'NG (19).jpg',
     'NG (190).jpg',
     'NG (191).jpg',
     'NG (192).jpg',
     'NG (193).jpg',
     'NG (194).jpg',
     'NG (195).jpg',
     'NG (196).jpg',
     'NG (197).jpg',
     'NG (198).jpg',
     'NG (199).jpg',
     'NG (2).jpg',
     'NG (20).jpg',
     'NG (200).jpg',
     'NG (201).jpg',
     'NG (202).jpg',
     'NG (203).jpg',
     'NG (204).jpg',
     'NG (205).jpg',
     'NG (206).jpg',
     'NG (207).jpg',
     'NG (208).jpg',
     'NG (209).jpg',
     'NG (21).jpg',
     'NG (210).jpg',
     'NG (211).jpg',
     'NG (212).jpg',
     'NG (213).jpg',
     'NG (214).jpg',
     'NG (215).jpg',
     'NG (216).jpg',
     'NG (217).jpg',
     'NG (218).jpg',
     'NG (219).jpg',
     'NG (22).jpg',
     'NG (220).jpg',
     'NG (221).jpg',
     'NG (222).jpg',
     'NG (223).jpg',
     'NG (224).jpg',
     'NG (225).jpg',
     'NG (226).jpg',
     'NG (227).jpg',
     'NG (228).jpg',
     'NG (229).jpg',
     'NG (23).jpg',
     'NG (230).jpg',
     'NG (231).jpg',
     'NG (232).jpg',
     'NG (233).jpg',
     'NG (234).jpg',
     'NG (235).jpg',
     'NG (236).jpg',
     'NG (237).jpg',
     'NG (238).jpg',
     'NG (239).jpg',
     'NG (24).jpg',
     'NG (240).jpg',
     'NG (241).jpg',
     'NG (242).jpg',
     'NG (243).jpg',
     'NG (244).jpg',
     'NG (245).jpg',
     'NG (246).jpg',
     'NG (247).jpg',
     'NG (248).jpg',
     'NG (249).jpg',
     'NG (25).jpg',
     'NG (250).jpg',
     'NG (251).jpg',
     'NG (252).jpg',
     'NG (253).jpg',
     'NG (254).jpg',
     'NG (255).jpg',
     'NG (256).jpg',
     'NG (257).jpg',
     'NG (258).jpg',
     'NG (259).jpg',
     'NG (26).jpg',
     'NG (260).jpg',
     'NG (261).jpg',
     'NG (262).jpg',
     'NG (263).jpg',
     'NG (264).jpg',
     'NG (265).jpg',
     'NG (266).jpg',
     'NG (267).jpg',
     'NG (268).jpg',
     'NG (269).jpg',
     'NG (27).jpg',
     'NG (270).jpg',
     'NG (271).jpg',
     'NG (272).jpg',
     'NG (273).jpg',
     'NG (274).jpg',
     'NG (275).jpg',
     'NG (276).jpg',
     'NG (277).jpg',
     'NG (278).jpg',
     'NG (279).jpg',
     'NG (28).jpg',
     'NG (280).jpg',
     'NG (281).jpg',
     'NG (282).jpg',
     'NG (283).jpg',
     'NG (284).jpg',
     'NG (285).jpg',
     'NG (286).jpg',
     'NG (287).jpg',
     'NG (288).jpg',
     'NG (289).jpg',
     'NG (29).jpg',
     'NG (290).jpg',
     'NG (291).jpg',
     'NG (292).jpg',
     'NG (293).jpg',
     'NG (294).jpg',
     'NG (295).jpg',
     'NG (296).jpg',
     'NG (297).jpg',
     'NG (298).jpg',
     'NG (299).jpg',
     'NG (3).jpg',
     'NG (30).jpg',
     'NG (300).jpg',
     'NG (301).jpg',
     'NG (302).jpg',
     'NG (303).jpg',
     'NG (304).jpg',
     'NG (305).jpg',
     'NG (306).jpg',
     'NG (307).jpg',
     'NG (308).jpg',
     'NG (309).jpg',
     'NG (31).jpg',
     'NG (310).jpg',
     'NG (311).jpg',
     'NG (312).jpg',
     'NG (313).jpg',
     'NG (314).jpg',
     'NG (315).jpg',
     'NG (316).jpg',
     'NG (317).jpg',
     'NG (318).jpg',
     'NG (319).jpg',
     'NG (32).jpg',
     'NG (320).jpg',
     'NG (321).jpg',
     'NG (322).jpg',
     'NG (323).jpg',
     'NG (324).jpg',
     'NG (325).jpg',
     'NG (326).jpg',
     'NG (327).jpg',
     'NG (328).jpg',
     'NG (329).jpg',
     'NG (33).jpg',
     'NG (330).jpg',
     'NG (331).jpg',
     'NG (332).jpg',
     'NG (333).jpg',
     'NG (334).jpg',
     'NG (335).jpg',
     'NG (336).jpg',
     'NG (337).jpg',
     'NG (338).jpg',
     'NG (339).jpg',
     'NG (34).jpg',
     'NG (340).jpg',
     'NG (341).jpg',
     'NG (342).jpg',
     'NG (343).jpg',
     'NG (344).jpg',
     'NG (345).jpg',
     'NG (346).jpg',
     'NG (347).jpg',
     'NG (348).jpg',
     'NG (349).jpg',
     'NG (35).jpg',
     'NG (350).jpg',
     'NG (351).jpg',
     'NG (352).jpg',
     'NG (353).jpg',
     'NG (354).jpg',
     'NG (355).jpg',
     'NG (356).jpg',
     'NG (357).jpg',
     'NG (358).jpg',
     'NG (359).jpg',
     'NG (36).jpg',
     'NG (360).jpg',
     'NG (361).jpg',
     'NG (362).jpg',
     'NG (363).jpg',
     'NG (364).jpg',
     'NG (365).jpg',
     'NG (366).jpg',
     'NG (367).jpg',
     'NG (368).jpg',
     'NG (369).jpg',
     'NG (37).jpg',
     'NG (370).jpg',
     'NG (371).jpg',
     'NG (372).jpg',
     'NG (373).jpg',
     'NG (374).jpg',
     'NG (375).jpg',
     'NG (376).jpg',
     'NG (377).jpg',
     'NG (378).jpg',
     'NG (379).jpg',
     'NG (38).jpg',
     'NG (380).jpg',
     'NG (381).jpg',
     'NG (382).jpg',
     'NG (383).jpg',
     'NG (384).jpg',
     'NG (385).jpg',
     'NG (386).jpg',
     'NG (387).jpg',
     'NG (388).jpg',
     'NG (389).jpg',
     'NG (39).jpg',
     'NG (390).jpg',
     'NG (391).jpg',
     'NG (392).jpg',
     'NG (393).jpg',
     'NG (394).jpg',
     'NG (395).jpg',
     'NG (396).jpg',
     'NG (397).jpg',
     'NG (398).jpg',
     'NG (399).jpg',
     'NG (4).jpg',
     'NG (40).jpg',
     'NG (400).jpg',
     'NG (401).jpg',
     'NG (402).jpg',
     'NG (403).jpg',
     'NG (404).jpg',
     'NG (405).jpg',
     'NG (406).jpg',
     'NG (407).jpg',
     'NG (408).jpg',
     'NG (409).jpg',
     'NG (41).jpg',
     'NG (410).jpg',
     'NG (411).jpg',
     'NG (412).jpg',
     'NG (413).jpg',
     'NG (414).jpg',
     'NG (415).jpg',
     'NG (416).jpg',
     'NG (417).jpg',
     'NG (418).jpg',
     'NG (419).jpg',
     'NG (42).jpg',
     'NG (420).jpg',
     'NG (421).jpg',
     'NG (422).jpg',
     'NG (423).jpg',
     'NG (424).jpg',
     'NG (425).jpg',
     'NG (426).jpg',
     'NG (427).jpg',
     'NG (428).jpg',
     'NG (429).jpg',
     'NG (43).jpg',
     'NG (430).jpg',
     'NG (431).jpg',
     'NG (432).jpg',
     'NG (433).jpg',
     'NG (434).jpg',
     'NG (435).jpg',
     'NG (436).jpg',
     'NG (437).jpg',
     'NG (438).jpg',
     'NG (439).jpg',
     'NG (44).jpg',
     'NG (440).jpg',
     'NG (441).jpg',
     'NG (442).jpg',
     'NG (443).jpg',
     'NG (444).jpg',
     'NG (445).jpg',
     'NG (446).jpg',
     'NG (447).jpg',
     'NG (448).jpg',
     'NG (449).jpg',
     'NG (45).jpg',
     'NG (450).jpg',
     'NG (451).jpg',
     'NG (452).jpg',
     'NG (453).jpg',
     'NG (454).jpg',
     'NG (455).jpg',
     'NG (456).jpg',
     'NG (457).jpg',
     'NG (458).jpg',
     'NG (459).jpg',
     'NG (46).jpg',
     'NG (460).jpg',
     'NG (461).jpg',
     'NG (462).jpg',
     'NG (463).jpg',
     'NG (464).jpg',
     'NG (465).jpg',
     'NG (466).jpg',
     'NG (467).jpg',
     'NG (468).jpg',
     'NG (469).jpg',
     'NG (47).jpg',
     'NG (470).jpg',
     'NG (471).jpg',
     'NG (472).jpg',
     'NG (473).jpg',
     'NG (474).jpg',
     'NG (475).jpg',
     'NG (476).jpg',
     'NG (477).jpg',
     'NG (478).jpg',
     'NG (479).jpg',
     'NG (48).jpg',
     'NG (480).jpg',
     'NG (481).jpg',
     'NG (482).jpg',
     'NG (483).jpg',
     'NG (484).jpg',
     'NG (485).jpg',
     'NG (486).jpg',
     'NG (487).jpg',
     'NG (488).jpg',
     'NG (489).jpg',
     'NG (49).jpg',
     'NG (490).jpg',
     'NG (491).jpg',
     'NG (492).jpg',
     'NG (493).jpg',
     'NG (494).jpg',
     'NG (495).jpg',
     'NG (496).jpg',
     'NG (497).jpg',
     'NG (498).jpg',
     'NG (499).jpg',
     'NG (5).jpg',
     'NG (50).jpg',
     'NG (500).jpg',
     'NG (51).jpg',
     'NG (52).jpg',
     'NG (53).jpg',
     'NG (54).jpg',
     'NG (55).jpg',
     'NG (56).jpg',
     'NG (57).jpg',
     'NG (58).jpg',
     'NG (59).jpg',
     'NG (6).jpg',
     'NG (60).jpg',
     'NG (61).jpg',
     'NG (62).jpg',
     'NG (63).jpg',
     'NG (64).jpg',
     'NG (65).jpg',
     'NG (66).jpg',
     'NG (67).jpg',
     'NG (68).jpg',
     'NG (69).jpg',
     'NG (7).jpg',
     'NG (70).jpg',
     'NG (71).jpg',
     'NG (72).jpg',
     'NG (73).jpg',
     'NG (74).jpg',
     'NG (75).jpg',
     'NG (76).jpg',
     'NG (77).jpg',
     'NG (78).jpg',
     'NG (79).jpg',
     'NG (8).jpg',
     'NG (80).jpg',
     'NG (81).jpg',
     'NG (82).jpg',
     'NG (83).jpg',
     'NG (84).jpg',
     'NG (85).jpg',
     'NG (86).jpg',
     'NG (87).jpg',
     'NG (88).jpg',
     'NG (89).jpg',
     'NG (9).jpg',
     'NG (90).jpg',
     'NG (91).jpg',
     'NG (92).jpg',
     'NG (93).jpg',
     'NG (94).jpg',
     'NG (95).jpg',
     'NG (96).jpg',
     'NG (97).jpg',
     'NG (98).jpg',
     'NG (99).jpg',
     'OK (1).jpg',
     'OK (10).jpg',
     'OK (100).jpg',
     'OK (101).jpg',
     'OK (102).jpg',
     'OK (103).jpg',
     'OK (104).jpg',
     'OK (105).jpg',
     'OK (106).jpg',
     'OK (107).jpg',
     'OK (108).jpg',
     'OK (109).jpg',
     'OK (11).jpg',
     'OK (110).jpg',
     'OK (111).jpg',
     'OK (112).jpg',
     'OK (113).jpg',
     'OK (114).jpg',
     'OK (115).jpg',
     'OK (116).jpg',
     'OK (117).jpg',
     'OK (118).jpg',
     'OK (119).jpg',
     'OK (12).jpg',
     'OK (120).jpg',
     'OK (121).jpg',
     'OK (122).jpg',
     'OK (123).jpg',
     'OK (124).jpg',
     'OK (125).jpg',
     'OK (126).jpg',
     'OK (127).jpg',
     'OK (128).jpg',
     'OK (129).jpg',
     'OK (13).jpg',
     'OK (130).jpg',
     'OK (131).jpg',
     'OK (132).jpg',
     'OK (133).jpg',
     'OK (134).jpg',
     'OK (135).jpg',
     'OK (136).jpg',
     'OK (137).jpg',
     'OK (138).jpg',
     'OK (139).jpg',
     'OK (14).jpg',
     'OK (140).jpg',
     'OK (141).jpg',
     'OK (142).jpg',
     'OK (143).jpg',
     'OK (144).jpg',
     'OK (145).jpg',
     'OK (146).jpg',
     'OK (147).jpg',
     'OK (148).jpg',
     'OK (149).jpg',
     'OK (15).jpg',
     'OK (150).jpg',
     'OK (151).jpg',
     'OK (152).jpg',
     'OK (153).jpg',
     'OK (154).jpg',
     'OK (155).jpg',
     'OK (156).jpg',
     'OK (157).jpg',
     'OK (158).jpg',
     'OK (159).jpg',
     'OK (16).jpg',
     'OK (160).jpg',
     'OK (161).jpg',
     'OK (162).jpg',
     'OK (163).jpg',
     'OK (164).jpg',
     'OK (165).jpg',
     'OK (166).jpg',
     'OK (167).jpg',
     'OK (168).jpg',
     'OK (169).jpg',
     'OK (17).jpg',
     'OK (170).jpg',
     'OK (171).jpg',
     'OK (172).jpg',
     'OK (173).jpg',
     'OK (174).jpg',
     'OK (175).jpg',
     'OK (176).jpg',
     'OK (177).jpg',
     'OK (178).jpg',
     'OK (179).jpg',
     'OK (18).jpg',
     'OK (180).jpg',
     'OK (181).jpg',
     'OK (182).jpg',
     'OK (183).jpg',
     'OK (184).jpg',
     'OK (185).jpg',
     'OK (186).jpg',
     'OK (187).jpg',
     'OK (188).jpg',
     'OK (189).jpg',
     'OK (19).jpg',
     'OK (190).jpg',
     'OK (191).jpg',
     'OK (192).jpg',
     'OK (193).jpg',
     'OK (194).jpg',
     'OK (195).jpg',
     'OK (196).jpg',
     'OK (197).jpg',
     'OK (198).jpg',
     'OK (199).jpg',
     'OK (2).jpg',
     'OK (20).jpg',
     'OK (200).jpg',
     'OK (201).jpg',
     'OK (202).jpg',
     'OK (203).jpg',
     'OK (204).jpg',
     'OK (205).jpg',
     'OK (206).jpg',
     'OK (207).jpg',
     'OK (208).jpg',
     'OK (209).jpg',
     'OK (21).jpg',
     'OK (210).jpg',
     'OK (211).jpg',
     'OK (212).jpg',
     'OK (213).jpg',
     'OK (214).jpg',
     'OK (215).jpg',
     'OK (216).jpg',
     'OK (217).jpg',
     'OK (218).jpg',
     'OK (219).jpg',
     'OK (22).jpg',
     'OK (220).jpg',
     'OK (221).jpg',
     'OK (222).jpg',
     'OK (223).jpg',
     'OK (224).jpg',
     'OK (225).jpg',
     'OK (226).jpg',
     'OK (227).jpg',
     'OK (228).jpg',
     'OK (229).jpg',
     'OK (23).jpg',
     'OK (230).jpg',
     'OK (231).jpg',
     'OK (232).jpg',
     'OK (233).jpg',
     'OK (234).jpg',
     'OK (235).jpg',
     'OK (236).jpg',
     'OK (237).jpg',
     'OK (238).jpg',
     'OK (239).jpg',
     'OK (24).jpg',
     'OK (240).jpg',
     'OK (241).jpg',
     'OK (242).jpg',
     'OK (243).jpg',
     'OK (244).jpg',
     'OK (245).jpg',
     'OK (246).jpg',
     'OK (247).jpg',
     'OK (248).jpg',
     'OK (249).jpg',
     'OK (25).jpg',
     'OK (250).jpg',
     'OK (251).jpg',
     'OK (252).jpg',
     'OK (253).jpg',
     'OK (254).jpg',
     'OK (255).jpg',
     'OK (256).jpg',
     'OK (257).jpg',
     'OK (258).jpg',
     'OK (259).jpg',
     'OK (26).jpg',
     'OK (260).jpg',
     'OK (261).jpg',
     'OK (262).jpg',
     'OK (263).jpg',
     'OK (264).jpg',
     'OK (265).jpg',
     'OK (266).jpg',
     'OK (267).jpg',
     'OK (268).jpg',
     'OK (269).jpg',
     'OK (27).jpg',
     'OK (270).jpg',
     'OK (271).jpg',
     'OK (272).jpg',
     'OK (273).jpg',
     'OK (274).jpg',
     'OK (275).jpg',
     'OK (276).jpg',
     'OK (277).jpg',
     'OK (278).jpg',
     'OK (279).jpg',
     'OK (28).jpg',
     'OK (280).jpg',
     'OK (281).jpg',
     'OK (282).jpg',
     'OK (283).jpg',
     'OK (284).jpg',
     'OK (285).jpg',
     'OK (286).jpg',
     'OK (287).jpg',
     'OK (288).jpg',
     'OK (289).jpg',
     'OK (29).jpg',
     'OK (290).jpg',
     'OK (291).jpg',
     'OK (292).jpg',
     'OK (293).jpg',
     'OK (294).jpg',
     'OK (295).jpg',
     'OK (296).jpg',
     'OK (297).jpg',
     'OK (298).jpg',
     'OK (299).jpg',
     'OK (3).jpg',
     'OK (30).jpg',
     'OK (300).jpg',
     'OK (301).jpg',
     'OK (302).jpg',
     'OK (303).jpg',
     'OK (304).jpg',
     'OK (305).jpg',
     'OK (306).jpg',
     'OK (307).jpg',
     'OK (308).jpg',
     'OK (309).jpg',
     'OK (31).jpg',
     'OK (310).jpg',
     'OK (311).jpg',
     'OK (312).jpg',
     'OK (313).jpg',
     'OK (314).jpg',
     'OK (315).jpg',
     'OK (316).jpg',
     'OK (317).jpg',
     'OK (318).jpg',
     'OK (319).jpg',
     'OK (32).jpg',
     'OK (320).jpg',
     'OK (321).jpg',
     'OK (322).jpg',
     'OK (323).jpg',
     'OK (324).jpg',
     'OK (325).jpg',
     'OK (326).jpg',
     'OK (327).jpg',
     'OK (328).jpg',
     'OK (329).jpg',
     'OK (33).jpg',
     'OK (330).jpg',
     'OK (331).jpg',
     'OK (332).jpg',
     'OK (333).jpg',
     'OK (334).jpg',
     'OK (335).jpg',
     'OK (336).jpg',
     'OK (337).jpg',
     'OK (338).jpg',
     'OK (339).jpg',
     'OK (34).jpg',
     'OK (340).jpg',
     'OK (341).jpg',
     'OK (342).jpg',
     'OK (343).jpg',
     'OK (344).jpg',
     'OK (345).jpg',
     'OK (346).jpg',
     'OK (347).jpg',
     'OK (348).jpg',
     'OK (349).jpg',
     'OK (35).jpg',
     'OK (350).jpg',
     'OK (351).jpg',
     'OK (352).jpg',
     'OK (353).jpg',
     'OK (354).jpg',
     'OK (355).jpg',
     'OK (356).jpg',
     'OK (357).jpg',
     'OK (358).jpg',
     'OK (359).jpg',
     'OK (36).jpg',
     'OK (360).jpg',
     'OK (361).jpg',
     'OK (362).jpg',
     'OK (363).jpg',
     'OK (364).jpg',
     'OK (365).jpg',
     'OK (366).jpg',
     'OK (367).jpg',
     'OK (368).jpg',
     'OK (369).jpg',
     'OK (37).jpg',
     'OK (370).jpg',
     'OK (371).jpg',
     'OK (372).jpg',
     'OK (373).jpg',
     'OK (374).jpg',
     'OK (375).jpg',
     'OK (376).jpg',
     'OK (377).jpg',
     'OK (378).jpg',
     'OK (379).jpg',
     'OK (38).jpg',
     'OK (380).jpg',
     'OK (381).jpg',
     'OK (382).jpg',
     'OK (383).jpg',
     'OK (384).jpg',
     'OK (385).jpg',
     'OK (386).jpg',
     'OK (387).jpg',
     'OK (388).jpg',
     'OK (389).jpg',
     'OK (39).jpg',
     'OK (390).jpg',
     'OK (391).jpg',
     'OK (392).jpg',
     'OK (393).jpg',
     'OK (394).jpg',
     'OK (395).jpg',
     'OK (396).jpg',
     'OK (397).jpg',
     'OK (398).jpg',
     'OK (399).jpg',
     'OK (4).jpg',
     'OK (40).jpg',
     'OK (400).jpg',
     'OK (401).jpg',
     'OK (402).jpg',
     'OK (403).jpg',
     'OK (404).jpg',
     'OK (405).jpg',
     'OK (406).jpg',
     'OK (407).jpg',
     'OK (408).jpg',
     'OK (409).jpg',
     'OK (41).jpg',
     'OK (410).jpg',
     'OK (411).jpg',
     'OK (412).jpg',
     'OK (413).jpg',
     'OK (414).jpg',
     'OK (415).jpg',
     'OK (416).jpg',
     'OK (417).jpg',
     'OK (418).jpg',
     'OK (419).jpg',
     'OK (42).jpg',
     'OK (420).jpg',
     'OK (421).jpg',
     'OK (422).jpg',
     'OK (423).jpg',
     'OK (424).jpg',
     'OK (425).jpg',
     'OK (426).jpg',
     'OK (427).jpg',
     'OK (428).jpg',
     'OK (429).jpg',
     'OK (43).jpg',
     'OK (430).jpg',
     'OK (431).jpg',
     'OK (432).jpg',
     'OK (433).jpg',
     'OK (434).jpg',
     'OK (435).jpg',
     'OK (436).jpg',
     'OK (437).jpg',
     'OK (438).jpg',
     'OK (439).jpg',
     'OK (44).jpg',
     'OK (440).jpg',
     'OK (441).jpg',
     'OK (442).jpg',
     'OK (443).jpg',
     'OK (444).jpg',
     'OK (445).jpg',
     'OK (446).jpg',
     'OK (447).jpg',
     'OK (448).jpg',
     'OK (449).jpg',
     'OK (45).jpg',
     'OK (450).jpg',
     'OK (451).jpg',
     'OK (452).jpg',
     'OK (453).jpg',
     'OK (454).jpg',
     'OK (455).jpg',
     'OK (456).jpg',
     'OK (457).jpg',
     'OK (458).jpg',
     'OK (459).jpg',
     'OK (46).jpg',
     'OK (460).jpg',
     'OK (461).jpg',
     'OK (462).jpg',
     'OK (463).jpg',
     'OK (464).jpg',
     'OK (465).jpg',
     'OK (466).jpg',
     'OK (467).jpg',
     'OK (468).jpg',
     'OK (469).jpg',
     'OK (47).jpg',
     'OK (470).jpg',
     'OK (471).jpg',
     'OK (472).jpg',
     'OK (473).jpg',
     'OK (474).jpg',
     'OK (475).jpg',
     'OK (476).jpg',
     'OK (477).jpg',
     'OK (478).jpg',
     'OK (479).jpg',
     'OK (48).jpg',
     'OK (480).jpg',
     'OK (481).jpg',
     'OK (482).jpg',
     'OK (483).jpg',
     'OK (484).jpg',
     'OK (485).jpg',
     'OK (486).jpg',
     'OK (487).jpg',
     'OK (488).jpg',
     'OK (489).jpg',
     'OK (49).jpg',
     'OK (490).jpg',
     'OK (491).jpg',
     'OK (492).jpg',
     'OK (493).jpg',
     'OK (494).jpg',
     'OK (495).jpg',
     'OK (496).jpg',
     'OK (497).jpg',
     'OK (498).jpg',
     'OK (499).jpg',
     'OK (5).jpg',
     'OK (50).jpg',
     'OK (500).jpg',
     'OK (501).jpg',
     'OK (502).jpg',
     'OK (503).jpg',
     'OK (504).jpg',
     'OK (505).jpg',
     'OK (506).jpg',
     'OK (507).jpg',
     'OK (508).jpg',
     'OK (509).jpg',
     'OK (51).jpg',
     'OK (510).jpg',
     'OK (511).jpg',
     'OK (512).jpg',
     'OK (513).jpg',
     'OK (514).jpg',
     'OK (515).jpg',
     'OK (516).jpg',
     'OK (517).jpg',
     'OK (518).jpg',
     'OK (519).jpg',
     'OK (52).jpg',
     'OK (520).jpg',
     'OK (521).jpg',
     'OK (522).jpg',
     'OK (523).jpg',
     'OK (524).jpg',
     'OK (525).jpg',
     'OK (526).jpg',
     'OK (527).jpg',
     'OK (528).jpg',
     'OK (529).jpg',
     'OK (53).jpg',
     'OK (530).jpg',
     'OK (531).jpg',
     'OK (532).jpg',
     'OK (533).jpg',
     'OK (534).jpg',
     'OK (535).jpg',
     'OK (536).jpg',
     'OK (537).jpg',
     'OK (538).jpg',
     'OK (539).jpg',
     'OK (54).jpg',
     'OK (540).jpg',
     'OK (541).jpg',
     'OK (542).jpg',
     'OK (543).jpg',
     'OK (544).jpg',
     'OK (545).jpg',
     'OK (546).jpg',
     'OK (547).jpg',
     'OK (548).jpg',
     'OK (549).jpg',
     ...]




```python
for filename in filenames:
    category = filename.split(' ')[0]
    if category == 'OK':
        categories.append(1)
    else:
        categories.append(0)
```


```python
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NG (1).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NG (10).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NG (100).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NG (101).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NG (102).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1175</th>
      <td>OK (95).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1176</th>
      <td>OK (96).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1177</th>
      <td>OK (97).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>OK (98).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1179</th>
      <td>OK (99).jpg</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1180 rows × 2 columns</p>
</div>




```python
df['category'].value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x282dd856b08>




![png](output_15_1.png)


## See sample image


```python
sample = random.choice(filenames)
sample
```




    'NG (376).jpg'




```python
image = load_img("D:/workspace/AOI/청북공장 이미지/train/"+sample)
```


```python
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x282df9535c8>




![png](output_19_1.png)


## Build Model


```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
```


```python
from tensorflow.keras.utils import plot_model
```


```python
model = Sequential()
```


```python
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
# filters : 필터(커널)의 개수,
# kernal_size : 필터의 크기
# activation : 활성화 함수
# input_shape : 첫 레이어에 인풋으로 들어오는 크기

# MaxPooling2D
# pool_size : 축소시킬 필터의 크기
# strides : 필터의 이동 간격, 기본값으로 pool_size를 가짐

# Dropout : rate 비율 만큼 drop 시킴
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # 1차원으로 변환하는 layer
model.add(Dense(512, activation='relu')) # unit: 완전연결 layer로 아웃풋 개수
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary() #모델의 layer와 속성 확인
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 234, 234, 32)      896       
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 234, 234, 32)      128       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 117, 117, 32)      0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 117, 117, 32)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 115, 115, 64)      18496     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 115, 115, 64)      256       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 57, 57, 64)        0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 57, 57, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 55, 55, 128)       73856     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 55, 55, 128)       512       
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 27, 27, 128)       0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 27, 27, 128)       0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 93312)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 512)               47776256  
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 512)               2048      
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 1026      
    =================================================================
    Total params: 47,873,474
    Trainable params: 47,872,002
    Non-trainable params: 1,472
    _________________________________________________________________
    

## Callbacks


```python
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
```

Early Stop

To prevent over fitting we will stop the learning after 10 epochs and val_loss value not decreased


```python
earlystop = EarlyStopping(patience=10)
```

Learning Rate Reducation

We will reduce the learning rate when then accuracy not increase for 2 steps


```python
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
```


```python
callbacks = [earlystop, learning_rate_reduction]
```

## Prepare Data


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NG (1).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NG (10).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NG (100).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NG (101).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NG (102).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1175</th>
      <td>OK (95).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1176</th>
      <td>OK (96).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1177</th>
      <td>OK (97).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>OK (98).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1179</th>
      <td>OK (99).jpg</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1180 rows × 2 columns</p>
</div>




```python
df["category"] = df["category"].replace({0: 'NG', 1: 'OK'}) 
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NG (1).jpg</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NG (10).jpg</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NG (100).jpg</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NG (101).jpg</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NG (102).jpg</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1175</th>
      <td>OK (95).jpg</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>1176</th>
      <td>OK (96).jpg</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>1177</th>
      <td>OK (97).jpg</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>OK (98).jpg</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>1179</th>
      <td>OK (99).jpg</td>
      <td>OK</td>
    </tr>
  </tbody>
</table>
<p>1180 rows × 2 columns</p>
</div>




```python
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
```


```python
train_df['category'].value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x282e6c7e148>




![png](output_39_1.png)



```python
validate_df['category'].value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x282e6ce1148>




![png](output_40_1.png)



```python
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15
```

## Training Generator


```python
batch_size
```




    15




```python
IMAGE_SIZE
```




    (236, 236)




```python
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "D:/workspace/AOI/청북공장 이미지/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
```

    Found 944 validated image filenames belonging to 2 classes.
    


```python
train_datagen
```




    <keras.preprocessing.image.ImageDataGenerator at 0x282e6d59988>



## Validation Generator


```python
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "D:/workspace/AOI/청북공장 이미지/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
```

    Found 236 validated image filenames belonging to 2 classes.
    

### See how our generator work


```python
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "D:/workspace/AOI/청북공장 이미지/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)
```

    Found 1 validated image filenames belonging to 1 classes.
    


```python
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
```


![png](output_51_0.png)


# Fit Model

#### epochs : 입력 데이터 학습 횟수
#### batch_size : 학습할 때 사용되는 데이터 수
#### verbose : 학습 중 출력되는 로그 수준(0,1,2)


```python
epochs=5 if FAST_RUN else 5
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)
```

    Epoch 1/5
    62/62 [==============================] - 124s 2s/step - loss: 0.9651 - accuracy: 0.7406 - val_loss: 3.8051 - val_accuracy: 0.6800
    Epoch 2/5
    

    C:\Users\ADMIN\anaconda3\envs\tf\lib\site-packages\keras\callbacks\callbacks.py:1042: RuntimeWarning: Reduce LR on plateau conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,val_accuracy,loss,accuracy,lr
      (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
    

    62/62 [==============================] - 128s 2s/step - loss: 0.6084 - accuracy: 0.7922 - val_loss: 0.2984 - val_accuracy: 0.7738
    Epoch 3/5
    62/62 [==============================] - 133s 2s/step - loss: 0.5312 - accuracy: 0.7922 - val_loss: 14.4431 - val_accuracy: 0.4344
    Epoch 4/5
    62/62 [==============================] - 135s 2s/step - loss: 0.4942 - accuracy: 0.8321 - val_loss: 0.1005 - val_accuracy: 0.8869
    Epoch 5/5
    62/62 [==============================] - 135s 2s/step - loss: 0.4254 - accuracy: 0.8418 - val_loss: 0.3092 - val_accuracy: 0.8643
    

Save Model


```python
model.save('model.h5')
model.save_weights("model.h5")
```

## Virtualize Training


```python
history.history
```




    {'val_loss': [3.805108070373535,
      0.2983675003051758,
      14.443097114562988,
      0.10052645206451416,
      0.30922383069992065],
     'val_accuracy': [0.6800000071525574,
      0.773755669593811,
      0.4343891441822052,
      0.8868778347969055,
      0.8642534017562866],
     'loss': [0.9653306498850893,
      0.6082466904422824,
      0.5314035771737443,
      0.4943541671471235,
      0.4253189745725035],
     'accuracy': [0.7405813, 0.79224974, 0.79224974, 0.8320775, 0.84176534],
     'lr': [0.001, 0.001, 0.001, 0.001, 0.001]}




```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
```


![png](output_59_0.png)


## preparing Test Data


```python
test_filenames = os.listdir("D:/workspace/AOI/청북공장 이미지/test")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
```


```python
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NG (1).jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NG (10).jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NG (10000).jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NG (10001).jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NG (10002).jpg</td>
    </tr>
  </tbody>
</table>
</div>



## Create Testing Generator


```python
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "D:/workspace/AOI/청북공장 이미지/test", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)
```

    Found 582 validated image filenames.
    

## Predict


```python
predict = model.predict_generator(test_generator, 
                                  steps=np.ceil(nb_samples/batch_size))
```

For categoral classication the prediction will come with probability of each category. So we will pick the category that have the highest probability with numpy average max


```python
test_df['category'] = np.argmax(predict, axis = -1)
```


```python
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NG (1).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NG (10).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NG (10000).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NG (10001).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NG (10002).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>577</th>
      <td>OK (95).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>578</th>
      <td>OK (96).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>579</th>
      <td>OK (97).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>580</th>
      <td>OK (98).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>581</th>
      <td>OK (99).jpg</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>582 rows × 2 columns</p>
</div>



We will convert the predict category back into our generator classes by using train_generator.class_indices. It is the classes that image generator map while converting data into computer vision




```python
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
```

From our prepare data part. We map data with {1: 'dog', 0: 'cat'}. Now we will map the result back to dog is 1 and cat is 0


```python
test_df['category'] = test_df['category'].replace({ 'NG': 0, 'OK': 1 })
```


```python
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NG (1).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NG (10).jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NG (10000).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NG (10001).jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NG (10002).jpg</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### virtualize result


```python
test_df['category'].value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x282e83e7048>




![png](output_76_1.png)



```python
IMAGE_SIZE
```




    (236, 236)




```python
sample_test = test_df.head(18)
sample_test.head()
#plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("D:/workspace/AOI/청북공장 이미지/test/"+filename, target_size=IMAGE_SIZE)
   # plt.subplot(4, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()
```


![png](output_78_0.png)



```python
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('S:/submission.csv', index=False)
```


```python
submission_df['real'] = submission_df.id.str.split(' ').str[0]
```


```python
submission_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
      <th>real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NG (1)</td>
      <td>0</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NG (10)</td>
      <td>0</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NG (10000)</td>
      <td>1</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NG (10001)</td>
      <td>1</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NG (10002)</td>
      <td>1</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>577</th>
      <td>OK (95)</td>
      <td>1</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>578</th>
      <td>OK (96)</td>
      <td>1</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>579</th>
      <td>OK (97)</td>
      <td>1</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>580</th>
      <td>OK (98)</td>
      <td>1</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>581</th>
      <td>OK (99)</td>
      <td>1</td>
      <td>OK</td>
    </tr>
  </tbody>
</table>
<p>582 rows × 3 columns</p>
</div>



accuracy print


```python
score = model.evaluate_generator(validation_generator,steps=len(validation_generator))
print('Test score:', score[0])
print('Test accuracy:', score[1])
```

    Test score: 0.2355787307024002
    Test accuracy: 0.8601694703102112
    

### confusion matrix


```python
from sklearn.metrics import confusion_matrix
```


```python
pd.crosstab(submission_df['real'], submission_df['label'], rownames=['True'], colnames=['Predicted'], margins=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NG</th>
      <td>182</td>
      <td>106</td>
      <td>288</td>
    </tr>
    <tr>
      <th>OK</th>
      <td>6</td>
      <td>288</td>
      <td>294</td>
    </tr>
    <tr>
      <th>All</th>
      <td>188</td>
      <td>394</td>
      <td>582</td>
    </tr>
  </tbody>
</table>
</div>



# 모델 불러오기


```python
from keras.models import load_model
```


```python
model.save('model.h5')
```


```python
json_string = model.to_json()
```


```python
model = load_model('model.h5')
```


```python
model_ss = tf.keras.models.load_model('model_ss.h5')
```


```python
model
```




    <keras.engine.sequential.Sequential at 0x282e7fb51c8>




```python
model_ss
```




    <tensorflow.python.keras.engine.sequential.Sequential at 0x28280258e08>




```python
import matplotlib.pyplot as plt
```


```python
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
```


```python
#tmp_name = "D:/workspace/AOI/청북공장 이미지/train_ver2/NG (8).jpg"
tmp_name = "D:/workspace/AOI/청북공장 이미지/OK_all/01000.jpg"

test_image = image.load_img(tmp_name, target_size = (236, 236))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
```


```python
test_gen = ImageDataGenerator(rescale=1./255)
```


```python
test_gen
```




    <keras.preprocessing.image.ImageDataGenerator at 0x2828f111d88>




```python
#filename : 데이터 들어오는 형식 보고 파일명=위치가 되도록 수정

b_filenames = os.listdir("D:/workspace/AOI/청북공장 이미지/barcode")
b_df = pd.DataFrame({
    'filename': b_filenames
})
nb_samples = b_df.shape[0]
```


```python
nb_samples
```




    6




```python
test_generator = test_gen.flow_from_dataframe(
    b_df, 
    "D:/workspace/AOI/청북공장 이미지/barcode", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)
```

    Found 6 validated image filenames.
    


```python
test_generator
```




    <keras.preprocessing.image.DataFrameIterator at 0x2828f10b288>




```python
test_generator
```




    <keras.preprocessing.image.DataFrameIterator at 0x2828f10b288>




```python
tmp_predict = model_ss.predict(test_generator, 
                                          steps=np.ceil(nb_samples/batch_size))
```


```python
tmp_predict
```




    array([[9.7499430e-01, 2.5005775e-02],
           [9.9996829e-01, 3.1740681e-05],
           [2.4090555e-01, 7.5909442e-01],
           [3.2524329e-01, 6.7475665e-01],
           [9.4656396e-01, 5.3436000e-02],
           [9.2177421e-01, 7.8225747e-02]], dtype=float32)




```python
tmp_predict[:1,]
```




    array([[0.9749943 , 0.02500577]], dtype=float32)




```python
plt.imshow(load_img(tmp_name))
```




    <matplotlib.image.AxesImage at 0x2828f341bc8>




![png](output_108_1.png)



```python
tmp = model.predict(test_image)
print(tmp)
```

    [[0. 1.]]
    


```python
if np.argmax(tmp) > 0:
    print("OK")
else:
    print("NG")
```

    OK
    


```python
rs = np.argmax(tmp)
print(rs)
```

    1
    


```python
tmp
```




    array([[0., 1.]], dtype=float32)



# model graph


```python
import pydotplus
```


```python
pydotplus.find_graphviz()
```




    {'dot': 'C:\\Users\\ADMIN\\anaconda3\\envs\\tf\\Library\\bin\\graphviz\\dot.exe',
     'twopi': 'C:\\Users\\ADMIN\\anaconda3\\envs\\tf\\Library\\bin\\graphviz\\twopi.exe',
     'neato': 'C:\\Users\\ADMIN\\anaconda3\\envs\\tf\\Library\\bin\\graphviz\\neato.exe',
     'circo': 'C:\\Users\\ADMIN\\anaconda3\\envs\\tf\\Library\\bin\\graphviz\\circo.exe',
     'fdp': 'C:\\Users\\ADMIN\\anaconda3\\envs\\tf\\Library\\bin\\graphviz\\fdp.exe',
     'sfdp': 'C:\\Users\\ADMIN\\anaconda3\\envs\\tf\\Library\\bin\\graphviz\\sfdp.exe'}




```python
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot

%matplotlib inline
```


```python
import matplotlib.pyplot as plt
!pip install graphviz
```


```python
SVG(model_to_dot(model,show_shapes=True).create(prog='dot',format='svg'))
```




![svg](output_118_0.svg)




```python

```
