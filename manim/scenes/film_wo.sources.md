# 那个精，便是我：字的来源

`film_wo.py` 用到的每段文字。原文在 `film_wo_texts.py`，渲染时从那里读，每个字全片只放一次，`film_wo.py` 里有硬断言。

核对方式：先在浏览器里读原典页面，再把每一行和抓下来的原页逐行比对（NFC 规范化，去空白后做子串匹配），未匹配的行人工看页面。「核对」列写结果。

| 语种 | 作者 | 作品 | 来源 | 核对 | 全片放了多少字（放上笔画的） |
|---|---|---|---|---|---|
| 法语 | 夏尔·波德莱尔 Charles Baudelaire | 《恶之花》·邀游 L’Invitation au voyage | https://fr.wikisource.org/wiki/Les_Fleurs_du_mal_(1861)/L’Invitation_au_voyage | 38/38 行匹配 | 560 / 924（13） |
| 西班牙语 | 米格尔·德·塞万提斯 Miguel de Cervantes | 《堂吉诃德》第一部第一章 Don Quijote | https://www.gutenberg.org/cache/epub/2000/pg2000-images.html | 13/13 行匹配 | 617 / 2497（31） |
| 意大利语 | 但丁·阿利吉耶里 Dante Alighieri | 《神曲·地狱篇》第一歌 Inferno, Canto I | https://it.wikisource.org/wiki/Divina_Commedia/Inferno/Canto_I | 48/48 行匹配 | 788 / 1793（29） |
| 德语 | 约翰·沃尔夫冈·冯·歌德 Johann Wolfgang von Goethe | 《浪游者的夜歌》 Ein Gleiches / Wandrers Nachtlied | https://de.wikisource.org/wiki/Ein_Gleiches ; https://de.wikisource.org/wiki/Wandrers_Nachtlied | 16/16 行匹配 | 305 / 390（17） |
| 俄语 | 亚历山大·普希金 Александр Пушкин | 《我曾经爱过你》 Я вас любил | https://ru.wikisource.org/wiki/Я_вас_любил:_любовь_ещё,_быть_может_(Пушкин) | 8/8 行匹配 | 226 / 282（15） |
| 拉丁语 | 维吉尔 Vergilius | 《埃涅阿斯纪》卷一 Aeneis I | https://la.wikisource.org/wiki/Aeneis/Liber_I | 33/33 行匹配 | 672 / 1492（24） |
| 葡萄牙语 | 路易斯·德·卡蒙斯 Luís de Camões | 《卢西塔尼亚人之歌》第一章 Os Lusíadas I | https://pt.wikisource.org/wiki/Os_Lusíadas_(grafia_atualizada)/I | 48/48 行匹配 | 637 / 1765（21） |
| 古希腊语 | 荷马 Ὅμηρος | 《伊利亚特》第一卷 Ἰλιάς Α | https://el.wikisource.org/wiki/Ιλιάς/Α | 25/25 行匹配 | 559 / 1091（24） |
| 英语（1609 四开本拼写） | 威廉·莎士比亚 William Shakespeare | 十四行诗第十八首 Sonnet 18 | https://en.wikisource.org/wiki/Shake-speares_Sonnets,_Never_before_Imprinted/Sonnet_18 | 14/14 行匹配 | 508 / 637（26） |
| 汉语 | 李白 | 《将进酒》 | https://zh.wikisource.org/zh-hans/將進酒_(李白) | 12/12 行匹配（先去掉页面夹注的 一作/又作/还作 异文） | 188 / 190（10） |
| 阿拉伯语 | 乌姆鲁勒·盖斯 امرؤ القيس | 《悬诗》开篇 معلقة امرئ القيس | https://ar.wikisource.org/wiki/معلقة_امرئ_القيس | 24/24 行匹配 | 117 / 117（19） |
| 波斯语 | 哈菲兹 حافظ | 《诗颂集》第一首 غزل ۱ | https://ganjoor.net/hafez/ghazal/sh1 | 14/14 行匹配 | 113 / 116（9） |
| 希伯来语 | 《希伯来圣经》 | 《创世记》第一章 בראשית א | https://mechon-mamre.org/p/pt/pt0101.htm | 7/7 行匹配 | 71 / 71（6） |
| 梵语 | 《梨俱吠陀》 | 《梨俱吠陀》1.1 ऋग्वेदः सूक्तं १.१ | https://sa.wikisource.org/wiki/ऋग्वेदः_सूक्तं_१.१ | 9/9 行匹配 | 82 / 82（6） |
| 泰米尔语 | திருவள்ளுவர் 蒂鲁瓦鲁瓦尔 | 《古拉尔箴言》第一章 திருக்குறள், கடவுள் வாழ்த்து | https://ta.wikisource.org/wiki/திருக்குறள்,_மூலம்/கடவுள்_வாழ்த்து | 20/20 行匹配 | 58 / 70（6） |
| 日语 | 松尾芭蕉 | 俳句「古池や」 | https://ja.wikipedia.org/wiki/古池や蛙飛びこむ水の音 | 1/1 行匹配 | 11 / 11（11） |
| 日语 | 松尾芭蕉 | 《奥之细道》开篇 おくのほそ道 | https://ja.wikisource.org/wiki/おくのほそ道 | 2/2 行匹配 | 196 / 196（11） |
| 韩语 | 金素月 김소월 | 《진달래꽃》《초혼》（诗集《진달래꽃》，1925） | https://ko.wikisource.org/wiki/진달래꽃_(시집)/진달래꽃 ; https://ko.wikisource.org/wiki/진달래꽃_(시집)/초혼 | 25/25 行匹配 | 289 / 375（13） |

## 字形

阿拉伯语、波斯语、希伯来语、梵语、泰米尔语由 HarfBuzz（渲染环境里已有的 hb-view / hb-shape）按词整形成图，再沿笔路摆放；其余语种按字母或方块字摆放。`python film_wo.py check` 对每种字体跑 hb-shape，断言没有 gid 0（缺字）。没有下载任何字体。

| 语种 | Studio 系统字体 |
|---|---|
| 阿拉伯语 | Muna Black |
| 波斯语 | Noto Nastaliq Urdu Bold |
| 希伯来语 | New Peninim MT Bold |
| 梵语 | Devanagari MT Bold |
| 泰米尔语 | Tamil MN Bold |

右行文字沿笔路阅读方向都是对的：笔路向左时词序从右到左；笔路向下时整行转向，词首在上。


## 沿用的文字

| 语种 | 作品 | 所用原文 | 来源 | 状态 |
|---|---|---|---|---|
| 阿卡德语楔形文字 | 《吉尔伽美什》标准版开篇 | 𒊭𒈾𒀝𒁀𒄿𒈬𒊒（Sha naqba īmuru） | 转写见 https://en.wikipedia.org/wiki/Epic_of_Gilgamesh | 转写本轮核对；符号序列沿用导演此前核对的版本，本轮未找到带 Unicode 楔形的原典页面。只有 7 个符号，全片各放一次 |
| 古埃及圣书体 | 《亡灵书》书名 rw nw prt m hrw，托勒密王名圈 | 𓂋𓏤𓈓𓏌𓏤𓉐𓂋𓏏𓂻𓅓𓉔𓂋𓅱𓇳𓏤 | 导演此前核对 | 本轮未重新核对，也未找到带 Unicode 圣书体的原典页面。16 个符号加一个王名圈，全片各放一次 |
| 甲骨文 | 八个单字 | 山鳥馬日月水木雨 | Wikimedia Commons（Richard Sears），字形数据只在 Studio，不进仓库 | 沿用 |
| 汉语 | 杜甫《春望》 | 國破山河在，城春草木深。感時花濺淚，恨別鳥驚心。烽火連三月，家書抵萬金。白頭搔更短，渾欲不勝簪。 | https://zh.wikisource.org/zh-hans/春望 | 正文逐字看过页面（页面为简体并夹注异文） |
| 汉语 | 陶渊明《与子俨等疏》（片中的家书） | 见 `film_docs.py` 的 LETTER | 维基文库，见 `film_docs.py` 注释 | 沿用 |
| 汉语 | 平政院裁决书（周树人诉章士钊，片中的裁决书） | 见 `film_docs.py` 的 RULING_* | 中国民商法律网，见 `film_docs.py` 注释 | 沿用 |
| 代码 | 欧几里得算法；K&R 的 hello, world | 见 `film_wo.py` 的 CODE | 片内 | 沿用 |
| 汉语 | 片中的帖子 | 见 `film_weibo.py` 的 POST，加末行「我是谁。」（`film_screens.py` 的 GROWN） | 片内虚构 | 沿用 |

## 所用原文

### 法语：夏尔·波德莱尔 Charles Baudelaire，《恶之花》·邀游 L’Invitation au voyage

https://fr.wikisource.org/wiki/Les_Fleurs_du_mal_(1861)/L’Invitation_au_voyage

原诗副歌（Là, tout n’est qu’ordre et beauté…）重复三次，这里只用第一次，后两次删去。

```text
Mon enfant, ma sœur,
Songe à la douceur
D’aller là-bas vivre ensemble !
Aimer à loisir,
Aimer et mourir
Au pays qui te ressemble !
Les soleils mouillés
De ces ciels brouillés
Pour mon esprit ont les charmes
Si mystérieux
De tes traîtres yeux,
Brillant à travers leurs larmes.
Là, tout n’est qu’ordre et beauté,
Luxe, calme et volupté.
Des meubles luisants,
Polis par les ans,
Décoreraient notre chambre ;
Les plus rares fleurs
Mêlant leurs odeurs
Aux vagues senteurs de l’ambre,
Les riches plafonds,
Les miroirs profonds,
La splendeur orientale,
Tout y parlerait
À l’âme en secret
Sa douce langue natale.
Vois sur ces canaux
Dormir ces vaisseaux
Dont l’humeur est vagabonde ;
C’est pour assouvir
Ton moindre désir
Qu’ils viennent du bout du monde.
— Les soleils couchants
Revêtent les champs,
Les canaux, la ville entière,
D’hyacinthe et d’or ;
Le monde s’endort
Dans une chaude lumière.
```

### 西班牙语：米格尔·德·塞万提斯 Miguel de Cervantes，《堂吉诃德》第一部第一章 Don Quijote

https://www.gutenberg.org/cache/epub/2000/pg2000-images.html

第一部第一章开头，连续选段。

```text
En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor.
Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda.
El resto della concluían sayo de velarte, calzas de velludo para las fiestas, con sus pantuflos de lo mesmo, y los días de entresemana se honraba con su vellorí de lo más fino.
Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera.
Frisaba la edad de nuestro hidalgo con los cincuenta años; era de complexión recia, seco de carnes, enjuto de rostro, gran madrugador y amigo de la caza.
Quieren decir que tenía el sobrenombre de Quijada, o Quesada, que en esto hay alguna diferencia en los autores que deste caso escriben; aunque, por conjeturas verosímiles, se deja entender que se llamaba Quejana.
Pero esto importa poco a nuestro cuento; basta que en la narración dél no se salga un punto de la verdad.
Es, pues, de saber que este sobredicho hidalgo, los ratos que estaba ocioso, que eran los más del año, se daba a leer libros de caballerías, con tanta afición y gusto, que olvidó casi de todo punto el ejercicio de la caza, y aun la administración de su hacienda.
Y llegó a tanto su curiosidad y desatino en esto, que vendió muchas hanegas de tierra de sembradura para comprar libros de caballerías en que leer, y así, llevó a su casa todos cuantos pudo haber dellos;
y de todos, ningunos le parecían tan bien como los que compuso el famoso Feliciano de Silva, porque la claridad de su prosa y aquellas entricadas razones suyas le parecían de perlas, y más cuando llegaba a leer aquellos requiebros y cartas de desafíos, donde en muchas partes hallaba escrito:
La razón de la sinrazón que a mi razón se hace, de tal manera mi razón enflaquece, que con razón me quejo de la vuestra fermosura.
Y también cuando leía: ...los altos cielos que de vuestra divinidad divinamente con las estrellas os fortifican, y os hacen merecedora del merecimiento que merece la vuestra grandeza.
Con estas razones perdía el pobre caballero el juicio, y desvelábase por entenderlas y desentrañarles el sentido, que no se lo sacara ni las entendiera el mesmo Aristóteles, si resucitara para sólo ello.
```

### 意大利语：但丁·阿利吉耶里 Dante Alighieri，《神曲·地狱篇》第一歌 Inferno, Canto I

https://it.wikisource.org/wiki/Divina_Commedia/Inferno/Canto_I

```text
Nel mezzo del cammin di nostra vita
mi ritrovai per una selva oscura,
ché la diritta via era smarrita.
Ahi quanto a dir qual era è cosa dura
esta selva selvaggia e aspra e forte
che nel pensier rinova la paura!
Tant’è amara che poco è più morte;
ma per trattar del ben ch’i’ vi trovai,
dirò de l’altre cose ch’i’ v’ ho scorte.
Io non so ben ridir com’i’ v’intrai,
tant’era pien di sonno a quel punto
che la verace via abbandonai.
Ma poi ch’i’ fui al piè d’un colle giunto,
là dove terminava quella valle
che m’avea di paura il cor compunto,
guardai in alto e vidi le sue spalle
vestite già de’ raggi del pianeta
che mena dritto altrui per ogne calle.
Allor fu la paura un poco queta,
che nel lago del cor m’era durata
la notte ch’i’ passai con tanta pieta.
E come quei che con lena affannata,
uscito fuor del pelago a la riva,
si volge a l’acqua perigliosa e guata,
così l’animo mio, ch’ancor fuggiva,
si volse a retro a rimirar lo passo
che non lasciò già mai persona viva.
Poi ch’èi posato un poco il corpo lasso,
ripresi via per la piaggia diserta,
sì che ’l piè fermo sempre era ’l più basso.
Ed ecco, quasi al cominciar de l’erta,
una lonza leggera e presta molto,
che di pel macolato era coverta;
e non mi si partia dinanzi al volto,
anzi ’mpediva tanto il mio cammino,
ch’i’ fui per ritornar più volte vòlto.
Temp’era dal principio del mattino,
e ’l sol montava ’n sù con quelle stelle
ch’eran con lui quando l’amor divino
mosse di prima quelle cose belle;
sì ch’a bene sperar m’era cagione
di quella fiera a la gaetta pelle
l’ora del tempo e la dolce stagione;
ma non sì che paura non mi desse
la vista che m’apparve d’un leone.
Questi parea che contra me venisse
con la test’alta e con rabbiosa fame,
sì che parea che l’aere ne tremesse.
```

### 德语：约翰·沃尔夫冈·冯·歌德 Johann Wolfgang von Goethe，《浪游者的夜歌》 Ein Gleiches / Wandrers Nachtlied

https://de.wikisource.org/wiki/Ein_Gleiches ; https://de.wikisource.org/wiki/Wandrers_Nachtlied

歌德两首同题诗，先 Ein Gleiches，后 Wandrers Nachtlied。

```text
Ueber allen Gipfeln
Ist Ruh’,
In allen Wipfeln
Spürest du
Kaum einen Hauch;
Die Vögelein schweigen im Walde.
Warte nur, balde
Ruhest du auch.
Der du von dem Himmel bist,
Alles Leid und Schmerzen stillest,
Den, der doppelt elend ist,
Doppelt mit Erquickung füllest,
Ach ich bin des Treibens müde!
Was soll all der Schmerz und Lust?
Süßer Friede,
Komm, ach komm in meine Brust!
```

### 俄语：亚历山大·普希金 Александр Пушкин，《我曾经爱过你》 Я вас любил

https://ru.wikisource.org/wiki/Я_вас_любил:_любовь_ещё,_быть_может_(Пушкин)

```text
Я вас любил: любовь ещё, быть может,
В душе моей угасла не совсем;
Но пусть она вас больше не тревожит;
Я не хочу печалить вас ничем.
Я вас любил безмолвно, безнадежно,
То робостью, то ревностью томим;
Я вас любил так искренно, так нежно,
Как дай вам Бог любимой быть другим.
```

### 拉丁语：维吉尔 Vergilius，《埃涅阿斯纪》卷一 Aeneis I

https://la.wikisource.org/wiki/Aeneis/Liber_I

```text
Arma virumque canō, Trōiae quī prīmus ab ōrīs
Ītaliam, fātō profugus, Lāvīniaque vēnit
lītora, multum ille et terrīs iactātus et altō
vī superum saevae memorem Iūnōnis ob īram;
multa quoque et bellō passus, dum conderet urbem,
īnferretque deōs Latiō, genus unde Latīnum,
Albānīque patrēs, atque altae moenia Rōmae.
Mūsa, mihī causās memorā, quō nūmine laesō,
quidve dolēns, rēgīna deum tot volvere cāsūs
īnsignem pietāte virum, tot adīre labōrēs
impulerit. Tantaene animīs caelestibus īrae?
Urbs antīqua fuit, Tyriī tenuēre colōnī,
Karthāgō, Ītaliam contrā Tiberīnaque longē
ōstia, dīves opum studiīsque asperrima bellī;
quam Iūnō fertur terrīs magis omnibus ūnam
posthabitā coluisse Samō; hīc illius arma,
hīc currus fuit; hoc rēgnum dea gentibus esse,
sī quā fāta sinant, iam tum tenditque fovetque.
'Prōgeniem' sed enim 'Trōiānō ā sanguine dūcī'
audierat 'Tyriās ōlim quae verteret arcēs;
hinc populum lātē rēgem bellōque superbum
ventūrum excidiō Libyae: sīc volvere Parcās.
Id metuēns, veterisque memor Sāturnia bellī,
prīma quod ad Trōiam prō cārīs gesserat Argīs
necdum etiam causae īrārum saevīque dolōrēs
exciderant animō: manet altā mente repostum
iūdicium Paridis sprētaeque iniūria fōrmae,
et genus invīsum, et raptī Ganymēdis honōrēs.
Hīs accēnsa super, iactātōs aequore tōtō
Trōas, reliquiās Danaum atque immītis Achillī,
arcēbat longē Latiō, multōsque per annōs
errābant, āctī fātīs, maria omnia circum.
Tantae mōlis erat Rōmānam condere gentem!
```

### 葡萄牙语：路易斯·德·卡蒙斯 Luís de Camões，《卢西塔尼亚人之歌》第一章 Os Lusíadas I

https://pt.wikisource.org/wiki/Os_Lusíadas_(grafia_atualizada)/I

```text
As armas e os barões assinalados,
Que da ocidental praia Lusitana,
Por mares nunca de antes navegados,
Passaram ainda além da Taprobana,
Em perigos e guerras esforçados,
Mais do que prometia a força humana,
E entre gente remota edificaram
Novo Reino, que tanto sublimaram;
E também as memórias gloriosas
Daqueles Reis, que foram dilatando
A Fé, o Império, e as terras viciosas
De África e de Ásia andaram devastando;
E aqueles, que por obras valerosas
Se vão da lei da morte libertando;
Cantando espalharei por toda parte,
Se a tanto me ajudar o engenho e arte.
Cessem do sábio Grego e do Troiano
As navegações grandes que fizeram;
Cale-se de Alexandro e de Trajano
A fama das vitórias que tiveram;
Que eu canto o peito ilustre Lusitano,
A quem Neptuno e Marte obedeceram:
Cesse tudo o que a Musa antiga canta,
Que outro valor mais alto se alevanta.
E vós, Tágides minhas, pois criado
Tendes em mim um novo engenho ardente,
Se sempre em verso humilde celebrado
Foi de mim vosso rio alegremente,
Dai-me agora um som alto e sublimado,
Um estilo grandíloquo e corrente,
Porque de vossas águas, Febo ordene
Que não tenham inveja às de Hipocrene.
Dai-me uma fúria grande e sonorosa,
E não de agreste avena ou frauta ruda,
Mas de tuba canora e belicosa,
Que o peito acende e a cor ao gesto muda;
Dai-me igual canto aos feitos da famosa
Gente vossa, que a Marte tanto ajuda;
Que se espalhe e se cante no universo,
Se tão sublime preço cabe em verso.
E vós, ó bem nascida segurança
Da Lusitana antiga liberdade,
E não menos certíssima esperança
De aumento da pequena Cristandade;
Vós, ó novo temor da Maura lança,
Maravilha fatal da nossa idade,
Dada ao mundo por Deus, que todo o mande,
Para do mundo a Deus dar parte grande;
```

### 古希腊语：荷马 Ὅμηρος，《伊利亚特》第一卷 Ἰλιάς Α

https://el.wikisource.org/wiki/Ιλιάς/Α

```text
Μῆνιν ἄειδε, θεά, Πηληϊάδεω Ἀχιλῆος
οὐλομένην, ἣ μυρί’ Ἀχαιοῖς ἄλγε’ ἔθηκε,
πολλὰς δ’ ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν
ἡρώων, αὐτοὺς δὲ ἑλώρια τεῦχε κύνεσσιν
οἰωνοῖσί τε πᾶσι· Διὸς δ’ ἐτελείετο βουλή,
ἐξ οὗ δὴ τὰ πρῶτα διαστήτην ἐρίσαντε
Ἀτρεΐδης τε ἄναξ ἀνδρῶν καὶ δῖος Ἀχιλλεύς.
Τίς τ’ ἄρ σφωε θεῶν ἔριδι ξυνέηκε μάχεσθαι;
Λητοῦς καὶ Διὸς υἱός· ὁ γὰρ βασιλῆϊ χολωθεὶς
νοῦσον ἀνὰ στρατὸν ὦρσε κακήν, ὀλέκοντο δὲ λαοί,
οὕνεκα τὸν Χρύσην ἠτίμασεν ἀρητῆρα
Ἀτρεΐδης· ὃ γὰρ ἦλθε θοὰς ἐπὶ νῆας Ἀχαιῶν
λυσόμενός τε θύγατρα φέρων τ’ ἀπερείσι’ ἄποινα,
στέμματ’ ἔχων ἐν χερσὶν ἑκηβόλου Ἀπόλλωνος
χρυσέῳ ἀνὰ σκήπτρῳ, καὶ λίσσετο πάντας Ἀχαιούς,
Ἀτρεΐδα δὲ μάλιστα δύω, κοσμήτορε λαῶν·
«Ἀτρεΐδαι τε καὶ ἄλλοι ἐϋκνήμιδες Ἀχαιοί,
ὑμῖν μὲν θεοὶ δοῖεν Ὀλύμπια δώματ’ ἔχοντες
ἐκπέρσαι Πριάμοιο πόλιν, εὖ δ’ οἴκαδ’ ἱκέσθαι·
παῖδα δ’ ἐμοὶ λύσαιτε φίλην, τὰ δ’ ἄποινα δέχεσθαι,
ἁζόμενοι Διὸς υἱὸν ἑκηβόλον Ἀπόλλωνα.»
Ἔνθ’ ἄλλοι μὲν πάντες ἐπευφήμησαν Ἀχαιοὶ
αἰδεῖσθαί θ’ ἱερῆα καὶ ἀγλαὰ δέχθαι ἄποινα·
ἀλλ’ οὐκ Ἀτρεΐδῃ Ἀγαμέμνονι ἥνδανε θυμῷ,
ἀλλὰ κακῶς ἀφίει, κρατερὸν δ’ ἐπὶ μῦθον ἔτελλε·
```

### 英语（1609 四开本拼写）：威廉·莎士比亚 William Shakespeare，十四行诗第十八首 Sonnet 18

https://en.wikisource.org/wiki/Shake-speares_Sonnets,_Never_before_Imprinted/Sonnet_18

1609 四开本拼写。页面首字是下沉大写 S 加 hall，这里连写为 Shall。第 13 行页面作 breath，第 14 行句末是逗号，都按页面。上一轮写的 breathe 和句号已改掉。

```text
Shall I compare thee to a Summers day?
Thou art more louely and more temperate:
Rough windes do shake the darling buds of Maie,
And Sommers lease hath all too short a date:
Sometime too hot the eye of heauen shines,
And often is his gold complexion dimm'd,
And euery faire from faire some-time declines,
By chance, or natures changing course vntrim'd:
But thy eternall Sommer shall not fade,
Nor loose possession of that faire thou ow'st,
Nor shall death brag thou wandr'st in his shade,
When in eternall lines to time thou grow'st,
So long as men can breath or eyes can see,
So long liues this, and this giues life to thee,
```

### 汉语：李白，《将进酒》

https://zh.wikisource.org/zh-hans/將進酒_(李白)

页面夹注异文（一作…），只取正文。「岑夫子，丹丘生。（将）进酒杯莫停。」一句页面带括号，略去。

```text
君不见，黄河之水天上来，奔流到海不复回！
君不见，高堂明镜悲白发，朝如青丝暮成雪！
人生得意须尽欢，莫使金樽空对月。
天生我材必有用，千金散尽还复来。
烹羊宰牛且为乐，会须一饮三百杯。
与君歌一曲，请君为我倾耳听。
钟鼓馔玉不足贵，但愿长醉不复醒。
古来圣贤皆寂寞，惟有饮者留其名。
陈王昔时宴平乐，斗酒十千恣欢谑。
主人何为言少钱？径须沽取对君酌。
五花马，千金裘。
呼儿将出换美酒，与尔同销万古愁。
```

### 阿拉伯语：乌姆鲁勒·盖斯 امرؤ القيس，《悬诗》开篇 معلقة امرئ القيس

https://ar.wikisource.org/wiki/معلقة_امرئ_القيس

页面原文带元音符号，保留。

```text
قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ ومَنْزِلِ
بِسِقْطِ اللِّوَى بَيْنَ الدَّخُول فَحَوْملِ
فَتُوْضِحَ فَالمِقْراةِ لمْ يَعْفُ رَسْمُها
لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وشَمْألِ
تَرَى بَعَرَ الأرْآمِ فِي عَرَصَاتِهَا
وَقِيْعَانِهَا كَأنَّهُ حَبُّ فُلْفُلِ
كَأنِّي غَدَاةَ البَيْنِ يَوْمَ تَحَمَّلُوا
لَدَى سَمُرَاتِ الحَيِّ نَاقِفُ حَنْظَلِ
وُقُوْفاً بِهَا صَحْبِي عَلَّي مَطِيَّهُمُ
يَقُوْلُوْنَ:لاَ تَهْلِكْ أَسَىً وَتَجَمَّلِ
وإِنَّ شِفائِي عَبْرَةٌ مُهْرَاقَةٌ
فَهَلْ عِنْدَ رَسْمٍ دَارِسٍ مِنْ مُعَوَّلِ؟
كَدَأْبِكَ مِنْ أُمِّ الحُوَيْرِثِ قَبْلَهَا
وَجَارَتِهَا أُمِّ الرَّبَابِ بِمَأْسَلِ
إِذَا قَامَتَا تَضَوَّعَ المِسْكُ مِنْهُمَا
نَسِيْمَ الصَّبَا جَاءَتْ بِرَيَّا القَرَنْفُلِ
فَفَاضَتْ دُمُوْعُ العَيْنِ مِنِّي صَبَابَةً
عَلَى النَّحْرِ حَتَّى بَلَّ دَمْعِي مِحْمَلِي
ألاَ رُبَّ يَوْمٍ لَكَ مِنْهُنَّ صَالِحٍ
وَلاَ سِيَّمَا يَوْمٌ بِدَارَةِ جُلْجُلِ
ويَوْمَ عَقَرْتُ لِلْعَذَارَي مَطِيَّتِي
فَيَا عَجَباً مِنْ كورها المُتَحَمَّلِ
فَظَلَّ العَذَارَى يَرْتَمِيْنَ بِلَحْمِهَا
وشَحْمٍ كَهُدَّابِ الدِّمَقْسِ المُفَتَّلِ
```

### 波斯语：哈菲兹 حافظ，《诗颂集》第一首 غزل ۱

https://ganjoor.net/hafez/ghazal/sh1

ghazal 1 全篇。保留页面的零宽不连字和元音符号。

```text
اَلا یا اَیُّهَا السّاقی اَدِرْ کَأسَاً و ناوِلْها
که عشق آسان نمود اوّل ولی افتاد مشکل‌ها
به بویِ نافه‌ای کآخِر صبا زان طُرّه بُگشاید
ز تابِ جَعدِ مشکینش چه خون افتاد در دل‌ها
مرا در منزلِ جانان چه امنِ عیش، چون هر دَم
جَرَس فریاد می‌دارد که بَربندید مَحمِل‌ها
به مِی سجّاده رنگین کُن گَرَت پیرِ مُغان گوید
که سالِک بی‌خبر نَبْوَد ز راه و رسمِ منزل‌ها
شبِ تاریک و بیمِ موج و گِردابی چنین هایل
کجا دانند حالِ ما سبک‌بارانِ ساحل‌ها؟
همهْ کارم ز خودکامی به بدنامی کشید آخر
نهان کِی مانَد آن رازی کَزو سازند مَحفل‌ها؟
حضوری گر همی خواهی از او غایب مشو حافظ
مَتیٰ ما تَلْقَ مَنْ تَهْویٰ دَعِ الدُّنْیا وَ اَهْمِلْها
```

### 希伯来语：《希伯来圣经》，《创世记》第一章 בראשית א

https://mechon-mamre.org/p/pt/pt0101.htm

第 1–7 节。经节字母和 {פ} 标记不是正文，略去。

```text
בְּרֵאשִׁית, בָּרָא אֱלֹהִים, אֵת הַשָּׁמַיִם, וְאֵת הָאָרֶץ.
וְהָאָרֶץ, הָיְתָה תֹהוּ וָבֹהוּ, וְחֹשֶׁךְ, עַל-פְּנֵי תְהוֹם; וְרוּחַ אֱלֹהִים, מְרַחֶפֶת עַל-פְּנֵי הַמָּיִם.
וַיֹּאמֶר אֱלֹהִים, יְהִי אוֹר; וַיְהִי-אוֹר.
וַיַּרְא אֱלֹהִים אֶת-הָאוֹר, כִּי-טוֹב; וַיַּבְדֵּל אֱלֹהִים, בֵּין הָאוֹר וּבֵין הַחֹשֶׁךְ.
וַיִּקְרָא אֱלֹהִים לָאוֹר יוֹם, וְלַחֹשֶׁךְ קָרָא לָיְלָה; וַיְהִי-עֶרֶב וַיְהִי-בֹקֶר, יוֹם אֶחָד.
וַיֹּאמֶר אֱלֹהִים, יְהִי רָקִיעַ בְּתוֹךְ הַמָּיִם, וִיהִי מַבְדִּיל, בֵּין מַיִם לָמָיִם.
וַיַּעַשׂ אֱלֹהִים, אֶת-הָרָקִיעַ, וַיַּבְדֵּל בֵּין הַמַּיִם אֲשֶׁר מִתַּחַת לָרָקִיעַ, וּבֵין הַמַּיִם אֲשֶׁר מֵעַל לָרָקִיעַ; וַיְהִי-כֵן.
```

### 梵语：《梨俱吠陀》，《梨俱吠陀》1.1 ऋग्वेदः सूक्तं १.१

https://sa.wikisource.org/wiki/ऋग्वेदः_सूक्तं_१.१

第 1–9 颂。页面开头的 ॐ 略去。

```text
अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम् । होतारं रत्नधातमम् ॥१॥
अग्निः पूर्वेभिरृषिभिरीड्यो नूतनैरुत । स देवाँ एह वक्षति ॥२॥
अग्निना रयिमश्नवत् पोषमेव दिवेदिवे । यशसं वीरवत्तमम् ॥३॥
अग्ने यं यज्ञमध्वरं विश्वतः परिभूरसि । स इद्देवेषु गच्छति ॥४॥
अग्निर्होता कविक्रतुः सत्यश्चित्रश्रवस्तमः । देवो देवेभिरा गमत् ॥५॥
यदङ्ग दाशुषे त्वमग्ने भद्रं करिष्यसि । तवेत् तत् सत्यमङ्गिरः ॥६॥
उप त्वाग्ने दिवेदिवे दोषावस्तर्धिया वयम् । नमो भरन्त एमसि ॥७॥
राजन्तमध्वराणां गोपामृतस्य दीदिविम् । वर्धमानं स्वे दमे ॥८॥
स नः पितेव सूनवेऽग्ने सूपायनो भव । सचस्वा नः स्वस्तये ॥९॥
```

### 泰米尔语：திருவள்ளுவர் 蒂鲁瓦鲁瓦尔，《古拉尔箴言》第一章 திருக்குறள், கடவுள் வாழ்த்து

https://ta.wikisource.org/wiki/திருக்குறள்,_மூலம்/கடவுள்_வாழ்த்து

第一章前十首。第一首「அகர முதல」即「所有文字以 A 为首」。经节编号略去。

```text
அகர முதல எழுத்தெல்லாம் ஆதி
பகவன் முதற்றே உலகு.
கற்றதனால் ஆய பயனென்கொல் வாலறிவன்
நற்றாள் தொழாஅர் எனின்.
மலர்மிசை ஏகினான் மாணடி சேர்ந்தார்
நிலமிசை நீடுவாழ் வார்.
வேண்டுதல் வேண்டாமை இலானடி சேர்ந்தார்க்கு
யாண்டும் இடும்பை இல.
இருள்சேர் இருவினையும் சேரா இறைவன்
பொருள்சேர் புகழ்புரிந்தார் மாட்டு.
பொறிவாயில் ஐந்தவித்தான் பொய்தீர் ஒழுக்க
நெறிநின்றார் நீடுவாழ் வார்.
தனக்குவமை இல்லாதான் தாள்சேர்ந்தார்க் கல்லால்
மனக்கவலை மாற்றல் அரிது.
அறவாழி அந்தணன் தாள்சேர்ந்தார்க் கல்லால்
பிறவாழி நீந்தல் அரிது.
கோளில் பொறியின் குணமிலவே எண்குணத்தான்
தாளை வணங்காத் தலை.
பிறவிப் பெருங்கடல் நீந்துவர் நீந்தார்
இறைவன் அடிசேரா தார்.
```

### 日语：松尾芭蕉，俳句「古池や」

https://ja.wikipedia.org/wiki/古池や蛙飛びこむ水の音

全句。

```text
古池や蛙飛びこむ水の音
```

### 日语：松尾芭蕉，《奥之细道》开篇 おくのほそ道

https://ja.wikisource.org/wiki/おくのほそ道

开头两段到「もゝひきの破れをつゞり」。此版本无标点，〳〵 和（すみか）按页面。

```text
月日は百代の過客にしてゆきかふ年も又旅人なり舟の上に生涯をうかべ馬の口とらへて老をむかふるものは日〳〵旅にして旅を栖（すみか）とす古人も多く旅に死せるあり
予もいづれの年よりか片雲の風にさそはれて漂泊の思ひやまず海浜にさすらへ去年の秋江上の破屋に蜘のふるすを拂ひてやゝ年もくれ春立る霞の空に白川の關越んとそゞろ神の物につきて心をくるはせ道祖神のまねきにあひて取物手につかずもゝひきの破れをつゞり
```

### 韩语：金素月 김소월，《진달래꽃》《초혼》（诗集《진달래꽃》，1925）

https://ko.wikisource.org/wiki/진달래꽃_(시집)/진달래꽃 ; https://ko.wikisource.org/wiki/진달래꽃_(시집)/초혼

《진달래꽃》取页面的现代语（현대어）部分，诗内的括号汉字按页面。全片同一句只出现一次，所以原诗的反复句只留第一次：《진달래꽃》末节的「나 보기가 역겨워 / 가실 때에는」，《초혼》的「사랑하던 그 사람이어!」（共 4 次）、「설움에 겹도록 부르노라.」（共 2 次）、「부르다가 내가 죽을 이름이어!」（共 2 次）。金素月卒于 1934 年。上一轮的李光洙《무정》因作者争议和对白体裁撤下。

```text
나 보기가 역겨워
가실 때에는
말없이 고이 보내드리오리다.
영변(寧邊)에 약산(藥山)
진달래꽃
아름따다 가실 길에 뿌리오리다.
가시는 걸음 걸음
놓인 그 꽃을
사뿐히 즈려밟고 가시옵소서.
죽어도 아니 눈물 흘리오리다.
산산이 부서진 이름이어!
허공중(虛空中)에 헤어진 이름이어!
불러도 주인(主人)없는 이름이어!
부르다가 내가 죽을 이름이어!
심중(心中)에 남아있는 말 한마디는
끝끝내 마저하지 못하였구나.
사랑하던 그 사람이어!
붉은 해는 서산(西山) 마루에 걸리었다.
사슴이의 무리도 슬피 운다.
떨어져 나가앉은 산 위에서
나는 그대의 이름을 부르노라.
설움에 겹도록 부르노라.
부르는 소리는 비껴가지만
하늘과 땅 사이가 너무 넓구나.
선 채로 이 자리에 돌이 되어도
```
