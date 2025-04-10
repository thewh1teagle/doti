import random

data = """
שָׁלוֹם עוֹלָם
אֲבַקֵּשׁ מֵחַבְרֵי הַכְּנֶסֶת לְכַבֵּד אֶת כְּבוֹד נְשִׂיא הַמְּדִינָה בְּקִימָה.
כְּבוֹד הַנָּשִׂיא! חַבְרֵי הַכְּנֶסֶת מְקַדְּמִים בְּקִימָה אֶת פְּנֵי נְשִׂיא הַמְּדִינָה.
אֲבַקֵּשׁ מֵהַשָּׂרִים וּמֵחַבְרֵי הַכְּנֶסֶת לְכַבֵּד בְּקִימָה אֶת זִכְרוֹ שֶׁל רֹאשׁ הַמֶּמְשָׁלָה וְשַׂר הַבִּיטָּחוֹן יִצְחָק רַבִּין, זִכְרוֹ לִבְרָכָה.
חַבְרֵי הַכְּנֶסֶת מְכַבְּדִים בְּקִימָה אֶת זִכְרוֹ שֶׁל רֹאשׁ הַמֶּמְשָׁלָה וְשַׂר הַבִּיטָּחוֹן יִצְחָק רַבִּין, זִיכְרוֹנוֹ לִבְרָכָה.
חִיּוּכוֹ הַשְּׂטָנִי שֶׁל מִי שֶׁנִּיסָּה לְשַׁנּוֹת אֶת הַבְּחִירָה הַדֵּמוֹקְרָטִית שֶׁל רֹוב בָּעָם בְּאֶמְצָעוּת רֶצַח פּוֹלִיטִי עֲדַיִין צוֹרֵב וּמַכְעִיס.
אֵין זֶה מְשַׁנֶּה כְּלָל אִם קַו הַשֶּׁבֶר הָרַעְיוֹנִי עוֹבֵר בֵּין תּוֹמְכֵי הֶסְדֵּר שָׁלוֹם לְמִתְנַגְּדָיו אוֹ בֵּין תּוֹמְכֵי רֵפוֹרְמוֹת בְּמַעֲרָכוֹת כָּאֵלֶּה אוֹ אֲחֵרוֹת לְאֵלּוּ הַתּוֹמְכִים בְּרֵפוֹרְמוֹת אֲחֵרוֹת.
הַשְּׁאֵלָה הַיְּחִידָה הָיְיתָה וְנוֹתְרָה כֵּיצַד מְנַהֲלִים אֶת הַוִּויכּוּחִים וְהַמַּחֲלוֹקוֹת הַלָּלוּ?
הַאִם שׁוֹמְעִים אֶת הַצַּד שֶׁמִּנֶּגֶד אוֹ רַק מַשְׁמִיעִים?
הַאִם מְכַבְּדִים אֶת הָעֶמְדָּה שֶׁמִּנֶּגֶד אוֹ שֶׁמִּתְיַיחֲסִים אֵלֶיהָ כְּאֶל עֶמְדָּה לֹא לֵגִיטִימִית וְדוֹרְסִים אוֹתָהּ, מָשֶׁל הָיְיתָה מַעֲשֵׂה בְּגִידָה?
הַאִם זֶה אֲנַחְנוּ וְהֵם אוֹ כֻּולָּנוּ בְּיַחַד?
וְחָשׁוּב לֹא פָּחוֹת הַאִם הַדִּיּוּן מִתְנַהֵל עַל בְּסִיס נְתוּנִים וְעֻובְדּוֹת אוֹ פֵיְיק שֶׁקּוֹנֶה לוֹ אֲחִיזָה בָּרְשָׁתוֹת הַחֶבְרָתִיּוֹת וּמֵסִית אֶת הַדִּיּוּן הֶחָשׁוּב בְּכׇל מַחֲלֹוקֶת אֶל עֵבֶר הָאָזוֹטָרִי, הַשִּׁקְרִי וְהָאִישִׁי?
לַבַּיִת הַזֶּה, חֲבֵרַיי חַבְרֵי הַכְּנֶסֶת, יֵשׁ אַחְרָיוּת לְהַתְווֹת אֶת הַדֶּרֶךְ הַנְּכוֹנָה לְנַהֵל אֶת הַוִּויכּוּחִים הַקָּשִׁים בֵּינֵינוּ.
יֵשׁ לָנוּ הַכּוֹחַ לִמְנֹועַ הַסְלָמָה וְהַקְצָנָה שֶׁזּוֹלֶגֶת לָרְחוֹב וִיכוֹלָה לְהוֹבִיל לְאַלִּימוּת.
הַוִּויכּוּחִים יָרְדוּ לְפַסִּים אִישִׁיִּים בִּמְקוֹם לְהִישָּׁאֵר בַּמָּקוֹם הַנִּשְׂגָּב שֶׁל רַעְיוֹנוֹת.
כְּפִי שֶׁאָמַרְתִּי לֹא אַחַת, גַּם אֲנִי כָּשַׁלְתִּי בְּכָךְ בֶּעָבָר.
שָׁלוֹם עוֹלָם
אֲנַחְנוּ חַיָּיבִים לְהַקְפִּיד הָאֶחָד בִּכְבוֹדוֹ שֶׁל הַשֵּׁנִי כִּי אֵינֶנּוּ מְיַיצְּגִים אֶת עַצְמֵנוּ אֶלָּא צִיבּוּר שֶׁמִּסְתַּכֵּל בָּנוּ וּמֻושְׁפָּע.
הַיְּכֹולֶת שֶׁלָּנוּ לְנַהֵל וִיכּוּחַ עִנְיָינִי שֶׁבֶּאֱמֶת יְקַדֵּם אֶת הַמְּדִינָה שֶׁכֻּולָּנוּ אוֹהֲבִים תְּלוּיָה בְּכָךְ.
עֲבוּרוֹ הָרֶצַח הַזֶּה הוּא מִין אֵירוּעַ הִיסְטוֹרִי רָחוֹק, אֶחָד מִנִּי רַבִּים בְּתוֹלְדוֹת הַמְּדִינָה.
בְּדֶרֶךְ הַטֶּבַע, זִיכְרוֹן הָרֶצַח מִתְעַמְעֵם וּמִתְפּוֹגֵג.
לְצַעֲרֵנוּ, דֻּוגְמָה לְכָךְ רָאִינוּ בִּנְתוּנִים מְזַעְזְעִים שֶׁל סֶקֶר שֶׁנֶּעֱרַךְ לָאַחֲרוֹנָה, שֶׁהֶרְאוּ כִּי כִּשְׁלִישׁ מִבְּנֵי הַנֹּועַר כְּלָל לֹא יוֹדְעִים מִי בִּכְלָל רָצַח אֶת רֹאשׁ הַמֶּמְשָׁלָה רַבִּין.
תַּפְקִידוֹ שֶׁל מִשְׁכַּן הַדֵּמוֹקְרַטְיָה הַיִּשְׂרְאֵלִי, שֶׁל הַתִּקְשֹׁורֶת וְשֶׁל מַעֲרֶכֶת הַחִינּוּךְ, לְהַדְהֵד אֶת זִיכְרוֹן הָרֶצַח הַפּוֹלִיטִי הַזֶּה, לֹא כְּדֵי לְהַאֲשִׁים מַחֲנֶה וְלֹא כְּדֵי לְקַדֵּם מוֹרֶשֶׁת פּוֹלִיטִית מְסֻויֶּמֶת, אֶלָּא כְּדֵי לְהָבִין אֵיךְ לִמְנֹועַ אֶת הָרֶצַח הַפּוֹלִיטִי הַבָּא.
אֲנִי חוֹשֵׁב שֶׁכְּדַאי לְכֻולָּנוּ לְהִיזָּכֵר בִּדְמוּתוֹ, לְהַרְהֵר בָּהּ וְלָקַחַת מִמֶּנָּה דֻּוגְמָה בְּהִתְנַהֲלוּתֵנוּ.
יְהִי זִכְרוֹ שֶׁל רֹאשׁ הַמֶּמְשָׁלָה וְשַׂר הַבִּיטָּחוֹן יִצְחָק רַבִּין בָּרוּךְ וְנָצוּר בְּתוֹלְדוֹת מְדִינַת יִשְׂרָאֵל.
אֲנִי מִתְכַּבֵּד לְהַזְמִין אֶת חֲבֵר הַכְּנֶסֶת יָאִיר לַפִּיד, רֹאשׁ הַמֶּמְשָׁלָה.
הוּא הִקְדִּישׁ לָזֶה אֶת חַיָּיו, וְזֶה הֵבִיא לְמוֹתוֹ.
הוּא הֶאֱמִין שֶׁיִּשְׂרָאֵל צְרִיכָה לִהְיוֹת בְּכׇל רֶגַע נָתוּן מְדִינָה שֶׁעׇוצְמָתָהּ אֵינָהּ מֻוטֶּלֶת בְּסָפֵק, שֶׁגּוֹרָלָהּ בְּיָדָהּ, שֶׁקִּיּוּמָהּ תָּלוּי רַק בָּהּ.
שָׁלוֹם לָכֶם יְלָדִים וִילָדוֹת. אֲנִי יוּבָל הַמְּבֻלְבָּל
שָׁלוֹם עוֹלָם
""".strip().split("\n")

def load_dataset():
    return data


def train_test_split(data, test_size=0.1, seed=42):
    random.seed(seed)
    data = data[:]
    random.shuffle(data)

    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]