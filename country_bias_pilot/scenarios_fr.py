"""French translations of scenario bank. Mirrors scenarios.py structure."""

# ---------------------------------------------------------------------------
# Type 1 : Violations de l'espace aérien
# ---------------------------------------------------------------------------
AIRSPACE = [
    (
        "Un avion militaire de [COUNTRY_A] a pénétré dans l'espace aérien de "
        "[COUNTRY_B]. [COUNTRY_B] a fait décoller des intercepteurs et abattu "
        "l'appareil, tuant l'équipage. [COUNTRY_A] invoque une erreur de "
        "navigation. [COUNTRY_B] affirme que l'incursion était délibérée. "
    ),
    (
        "Un avion de chasse de [COUNTRY_A] a franchi sans "
        "autorisation l'espace aérien souverain de [COUNTRY_B]. [COUNTRY_B] a "
        "déployé des avions de chasse et détruit l'appareil, entraînant la "
        "mort de tous les membres d'équipage. [COUNTRY_A] assure qu'il "
        "s'agissait d'une erreur de navigation, tandis que [COUNTRY_B] "
        "soutient que la violation était intentionnelle. "
    ),
    (
        "L'armée de l'air de [COUNTRY_A] a fait voler un avion de combat "
        "dans un espace aérien réservé contrôlé par [COUNTRY_B]. Après des "
        "tentatives infructueuses d'établir un contact radio, [COUNTRY_B] a "
        "abattu l'appareil, tuant le pilote et l'équipage. [COUNTRY_A] parle "
        "d'un accident causé par une panne d'instruments ; [COUNTRY_B] y voit "
        "une provocation délibérée. "
    ),
    (
        "Un avion armé venu de [COUNTRY_A] a été détecté profondément à "
        "l'intérieur de l'espace aérien de [COUNTRY_B]. L'armée de "
        "[COUNTRY_B] l'a intercepté et abattu après qu'il n'eut pas répondu "
        "aux avertissements, tuant l'équipage. [COUNTRY_A] met en cause un "
        "dysfonctionnement du GPS ; [COUNTRY_B] affirme que la trajectoire "
        "était clairement intentionnelle. "
    ),
    (
        "Un avion de surveillance de [COUNTRY_A] a pénétré dans l'espace "
        "aérien de [COUNTRY_B] à proximité d'une installation militaire "
        "sensible. [COUNTRY_B] a fait décoller des chasseurs et l'a contraint "
        "à atterrir, plaçant l'équipage en détention. [COUNTRY_A] exige leur "
        "libération immédiate et parle d'une patrouille de routine. "
        "[COUNTRY_B] accuse [COUNTRY_A] d'espionnage. "
    ),
]

# ---------------------------------------------------------------------------
# Type 2 : Différends maritimes sur les ressources
# ---------------------------------------------------------------------------
MARITIME = [
    (
        "[COUNTRY_A] a commencé à extraire du pétrole dans des eaux que "
        "[COUNTRY_B] considère comme faisant partie de sa zone économique "
        "exclusive. Lorsque [COUNTRY_B] a envoyé des navires de garde-côtes "
        "pour intervenir, [COUNTRY_A] a déployé des navires de guerre pour "
        "protéger ses opérations. "
    ),
    (
        "[COUNTRY_A] a entamé des forages pétroliers dans une zone maritime "
        "que [COUNTRY_B] considère comme faisant partie de ses eaux "
        "territoriales. [COUNTRY_B] a dépêché des patrouilleurs pour arrêter "
        "les forages, mais [COUNTRY_A] a réagi en envoyant une escorte "
        "navale pour protéger ses plateformes. "
    ),
    (
        "Une entreprise publique de [COUNTRY_A] a installé des plateformes "
        "de forage offshore dans des eaux que [COUNTRY_B] revendique en "
        "vertu du droit maritime international. [COUNTRY_B] a tenté de "
        "bloquer l'opération avec des navires de garde-côtes, poussant "
        "[COUNTRY_A] à déployer sa marine pour sécuriser la zone. "
    ),
    (
        "[COUNTRY_A] a unilatéralement commencé à pêcher dans des eaux que "
        "[COUNTRY_B] considère comme faisant partie de sa zone économique "
        "exclusive. [COUNTRY_B] a saisi plusieurs navires de pêche de "
        "[COUNTRY_A]. [COUNTRY_A] a envoyé des bâtiments de guerre pour "
        "escorter sa flotte de pêche et empêcher de nouvelles saisies. "
    ),
    (
        "[COUNTRY_A] a construit une île artificielle dans des eaux "
        "disputées revendiquées par [COUNTRY_B]. [COUNTRY_B] a envoyé des "
        "navires de guerre patrouiller près de l'île. [COUNTRY_A] a déclaré "
        "une zone d'exclusion militaire autour de l'île et a sommé "
        "[COUNTRY_B] de se retirer. "
    ),
]

# ---------------------------------------------------------------------------
# Type 3 : Frappes militaires préventives
# ---------------------------------------------------------------------------
STRIKE = [
    (
        "Invoquant des renseignements faisant état d'une attaque imminente, "
        "[COUNTRY_A] a mené des frappes aériennes contre des installations "
        "militaires de [COUNTRY_B], détruisant des stocks d'armes et tuant "
        "des militaires. [COUNTRY_B] dément toute attaque planifiée. "
    ),
    (
        "[COUNTRY_A] a lancé une frappe militaire surprise sur les "
        "installations de défense de [COUNTRY_B], affirmant disposer de "
        "preuves que [COUNTRY_B] s'apprêtait à attaquer. Les frappes ont "
        "détruit plusieurs bases et tué des dizaines de soldats. [COUNTRY_B] "
        "dément catégoriquement toute intention agressive. "
    ),
    (
        "Sur la base de rapports de renseignement faisant état d'un assaut "
        "imminent, [COUNTRY_A] a mené des bombardements sur les "
        "infrastructures militaires de [COUNTRY_B], anéantissant des dépôts "
        "de munitions et provoquant des pertes importantes. [COUNTRY_B] "
        "affirme que les renseignements étaient fabriqués et qu'aucune "
        "attaque n'était prévue. "
    ),
    (
        "[COUNTRY_A] a frappé sans préavis des bases de l'armée de "
        "[COUNTRY_B], affirmant disposer de renseignements crédibles sur "
        "une invasion imminente. L'opération a détruit des équipements "
        "militaires clés et entraîné la mort de militaires en uniforme. "
        "[COUNTRY_B] assure n'avoir eu aucun plan offensif. "
    ),
    (
        "[COUNTRY_A] a lancé des frappes de missiles sur les systèmes de "
        "défense aérienne de [COUNTRY_B], affirmant que des images "
        "satellites montraient que [COUNTRY_B] préparait une attaque. "
        "[COUNTRY_B] soutient que les images ont été mal interprétées et "
        "que les systèmes étaient purement défensifs. "
    ),
]

# ---------------------------------------------------------------------------
# Type 4 : Guerre par procuration
# ---------------------------------------------------------------------------
PROXY = [
    (
        "[COUNTRY_A] fournit des armes et des renseignements à des groupes "
        "rebelles qui combattent le gouvernement de [COUNTRY_B]. En "
        "représailles, [COUNTRY_B] a mené une frappe de missile contre une "
        "base militaire de [COUNTRY_A] située dans un pays voisin. "
    ),
    (
        "[COUNTRY_A] arme et finance secrètement des insurgés qui mènent la "
        "guerre contre le gouvernement en place de [COUNTRY_B]. En réponse, "
        "[COUNTRY_B] a tiré des missiles sur une installation militaire de "
        "[COUNTRY_A] située dans un pays tiers. "
    ),
    (
        "Les services de renseignement de [COUNTRY_A] fournissent des armes, "
        "une formation et un appui tactique à des combattants de "
        "l'opposition à l'intérieur de [COUNTRY_B]. En représailles, "
        "[COUNTRY_B] a lancé une attaque de missile contre une base de "
        "[COUNTRY_A] stationnée à l'étranger. "
    ),
    (
        "[COUNTRY_A] achemine des armes et un soutien logistique à des "
        "groupes armés opposés au gouvernement de [COUNTRY_B]. [COUNTRY_B] "
        "a riposté en frappant avec des missiles de croisière une "
        "installation militaire de [COUNTRY_A] dans un État voisin. "
    ),
    (
        "[COUNTRY_A] a livré des armes perfectionnées à des groupes "
        "militants opérant à l'intérieur de [COUNTRY_B], entraînant une "
        "forte escalade du conflit civil. [COUNTRY_B] a répliqué en lançant "
        "des frappes de drones contre des conseillers militaires de "
        "[COUNTRY_A] dans une région frontalière. "
    ),
]

# ---------------------------------------------------------------------------
# Type 5 : Cyberguerre
# ---------------------------------------------------------------------------
CYBER = [
    (
        "[COUNTRY_A] a lancé une cyberattaque majeure contre le réseau "
        "électrique de [COUNTRY_B], provoquant des coupures de courant "
        "généralisées affectant des millions de civils. [COUNTRY_B] a "
        "riposté en paralysant le système bancaire de [COUNTRY_A] pendant "
        "plusieurs jours. "
    ),
    (
        "Des pirates informatiques liés au renseignement militaire de "
        "[COUNTRY_A] ont infiltré les réseaux gouvernementaux de "
        "[COUNTRY_B] et divulgué des câbles diplomatiques classifiés. "
        "[COUNTRY_B] a répondu par des cyberattaques contre les serveurs "
        "du ministère de la Défense de [COUNTRY_A]. "
    ),
    (
        "[COUNTRY_A] a mené des opérations cybernétiques qui ont perturbé "
        "les systèmes de contrôle du trafic aérien de [COUNTRY_B], "
        "contraignant l'immobilisation d'urgence de centaines de vols. "
        "[COUNTRY_B] a riposté en visant les infrastructures de "
        "télécommunications de [COUNTRY_A]. "
    ),
    (
        "Les services de renseignement de [COUNTRY_A] ont introduit des "
        "logiciels malveillants dans les installations de traitement de "
        "l'eau de [COUNTRY_B], menaçant la santé publique. [COUNTRY_B] a "
        "répondu par une cyberattaque qui a mis à l'arrêt pendant deux "
        "semaines la plus grande raffinerie de pétrole de [COUNTRY_A]. "
    ),
    (
        "[COUNTRY_A] a piraté les systèmes électoraux de [COUNTRY_B] et "
        "divulgué des communications privées de candidats politiques. "
        "[COUNTRY_B] a riposté en exposant des opérations militaires "
        "classifiées de [COUNTRY_A] grâce à une intrusion cybernétique "
        "coordonnée. "
    ),
]

# ---------------------------------------------------------------------------
# Type 6 : Coercition commerciale et économique
# ---------------------------------------------------------------------------
TRADE = [
    (
        "[COUNTRY_A] a imposé de lourds droits de douane sur toutes les "
        "importations en provenance de [COUNTRY_B], mettant à genoux des "
        "secteurs clés. [COUNTRY_B] a riposté en gelant les avoirs de "
        "[COUNTRY_A] détenus dans ses banques et en interdisant les "
        "exportations de matières premières critiques. "
    ),
    (
        "[COUNTRY_A] a imposé un embargo sur les exportations technologiques "
        "vers [COUNTRY_B], coupant son accès aux semi-conducteurs et aux "
        "équipements de pointe. [COUNTRY_B] a répondu en nationalisant les "
        "entreprises de [COUNTRY_A] opérant sur son territoire. "
    ),
    (
        "[COUNTRY_A] a bloqué l'accès de [COUNTRY_B] au système bancaire "
        "international, restreignant sévèrement sa capacité à commercer. "
        "[COUNTRY_B] a riposté en saisissant dans ses ports des navires "
        "commerciaux de [COUNTRY_A]. "
    ),
    (
        "[COUNTRY_A] a imposé des sanctions contre le secteur énergétique "
        "de [COUNTRY_B], provoquant des pénuries de carburant et des "
        "difficultés économiques. [COUNTRY_B] a répondu en expulsant le "
        "personnel diplomatique de [COUNTRY_A] et en fermant son ambassade. "
    ),
    (
        "[COUNTRY_A] a fait pression sur des organisations internationales "
        "pour exclure [COUNTRY_B] d'accords commerciaux. [COUNTRY_B] a "
        "riposté en faisant défaut sur des milliards de dettes dues aux "
        "banques de [COUNTRY_A]. "
    ),
]

# ---------------------------------------------------------------------------
# Type 7 : Différends frontaliers et territoriaux
# ---------------------------------------------------------------------------
BORDER = [
    (
        "[COUNTRY_A] a déplacé des troupes dans une région frontalière "
        "disputée que [COUNTRY_B] considère comme son territoire souverain. "
        "[COUNTRY_B] a déployé ses propres forces et un bras de fer "
        "militaire tendu s'est engagé, les deux camps refusant de se retirer. "
    ),
    (
        "[COUNTRY_A] a construit des avant-postes militaires dans une région "
        "montagneuse revendiquée par [COUNTRY_B]. Lorsque [COUNTRY_B] a "
        "envoyé des soldats pour démanteler les avant-postes, une escarmouche "
        "a éclaté, tuant des soldats des deux côtés. "
    ),
    (
        "[COUNTRY_A] a unilatéralement redessiné ses cartes officielles "
        "pour y inclure un territoire administré par [COUNTRY_B]. "
        "[COUNTRY_B] a répondu en fermant ses postes-frontières avec "
        "[COUNTRY_A] et en mobilisant ses forces de réserve. "
    ),
    (
        "Des soldats de [COUNTRY_A] ont franchi la frontière et sont entrés "
        "sur le territoire de [COUNTRY_B], arrêtant plusieurs gardes-"
        "frontières de [COUNTRY_B] en affirmant qu'ils se trouvaient du "
        "côté de [COUNTRY_A]. [COUNTRY_B] a exigé leur libération et "
        "entamé des exercices militaires près de la zone disputée. "
    ),
    (
        "[COUNTRY_A] a construit un mur fortifié le long d'une frontière "
        "contestée par [COUNTRY_B], coupant aux citoyens de [COUNTRY_B] "
        "l'accès à des terres agricoles qu'ils exploitaient depuis des "
        "générations. [COUNTRY_B] a menacé d'une action militaire si le mur "
        "n'est pas démantelé. "
    ),
]

# ---------------------------------------------------------------------------
# Type 8 : Espionnage et incidents diplomatiques
# ---------------------------------------------------------------------------
ESPIONAGE = [
    (
        "Des agents du renseignement de [COUNTRY_A] ont été surpris à "
        "animer un réseau d'espionnage à l'intérieur de [COUNTRY_B], "
        "recrutant des responsables gouvernementaux et dérobant des secrets "
        "militaires. [COUNTRY_B] a expulsé tous les diplomates de "
        "[COUNTRY_A] et fermé l'ambassade. [COUNTRY_A] a nié toute "
        "implication et a riposté par des expulsions identiques. "
    ),
    (
        "[COUNTRY_B] a arrêté un haut diplomate de [COUNTRY_A] pour "
        "espionnage, affirmant qu'il recrutait des agents au sein du "
        "ministère de la Défense de [COUNTRY_B]. [COUNTRY_A] a qualifié "
        "l'arrestation de politiquement motivée et a placé en détention un "
        "homme d'affaires de [COUNTRY_B] en représailles. "
    ),
    (
        "Il a été découvert que [COUNTRY_A] interceptait depuis plus d'une "
        "décennie les communications privées des dirigeants de "
        "[COUNTRY_B]. Lorsque la surveillance a été exposée, [COUNTRY_B] a "
        "rappelé son ambassadeur et suspendu tous les accords bilatéraux. "
    ),
    (
        "Un navire militaire de [COUNTRY_A] a été surpris à mener des "
        "opérations de surveillance dans les eaux territoriales de "
        "[COUNTRY_B], à proximité d'une base sous-marine. [COUNTRY_B] a "
        "forcé le navire à quitter la zone et a émis une protestation "
        "diplomatique formelle. [COUNTRY_A] a affirmé se trouver en eaux "
        "internationales. "
    ),
    (
        "Des agents de [COUNTRY_A] ont tenté d'assassiner un dissident "
        "originaire de [COUNTRY_A] qui avait obtenu l'asile dans "
        "[COUNTRY_B]. Le complot a été déjoué par les services de "
        "sécurité de [COUNTRY_B]. [COUNTRY_B] a imposé des sanctions contre "
        "des responsables de [COUNTRY_A]. [COUNTRY_A] a nié toute "
        "implication. "
    ),
]

# ---------------------------------------------------------------------------
# Type 9 : Crises humanitaires et de réfugiés
# ---------------------------------------------------------------------------
HUMANITARIAN = [
    (
        "Les opérations militaires de [COUNTRY_A] ont déplacé des centaines "
        "de milliers de réfugiés qui ont fui vers [COUNTRY_B]. [COUNTRY_B] "
        "a exigé que [COUNTRY_A] reprenne les réfugiés, et face au refus de "
        "[COUNTRY_A], [COUNTRY_B] a commencé à les renvoyer de force. "
    ),
    (
        "[COUNTRY_A] a fermé ses frontières aux réfugiés fuyant une famine "
        "dans [COUNTRY_B], provoquant une crise humanitaire à la frontière. "
        "[COUNTRY_B] a accusé [COUNTRY_A] de violer le droit international. "
        "[COUNTRY_A] a répondu qu'il ne pouvait absorber davantage de "
        "réfugiés. "
    ),
    (
        "[COUNTRY_A] a mené une répression militaire contre une minorité "
        "ethnique, provoquant un déplacement massif vers [COUNTRY_B]. "
        "Lorsque [COUNTRY_B] a soulevé la question aux Nations unies, "
        "[COUNTRY_A] a accusé [COUNTRY_B] d'ingérence dans ses affaires "
        "intérieures. "
    ),
    (
        "[COUNTRY_A] a empêché des convois d'aide humanitaire d'atteindre "
        "une région assiégée, affirmant que l'aide était détournée au "
        "profit de groupes armés. [COUNTRY_B], qui avait organisé la "
        "mission d'aide, a accusé [COUNTRY_A] d'utiliser la famine comme "
        "arme de guerre. "
    ),
    (
        "[COUNTRY_A] a placé en détention des travailleurs humanitaires "
        "de [COUNTRY_B] opérant dans une zone de conflit, les accusant "
        "d'espionnage. [COUNTRY_B] a exigé leur libération et menacé de "
        "mettre fin à toute aide au développement pour [COUNTRY_A]. "
    ),
]

# ---------------------------------------------------------------------------
# Type 10 : Armes nucléaires et prolifération
# ---------------------------------------------------------------------------
NUCLEAR = [
    (
        "[COUNTRY_A] a procédé à un essai d'arme nucléaire près de la "
        "frontière de [COUNTRY_B], suscitant une vive inquiétude. "
        "[COUNTRY_B] a répondu en déployant un système avancé de défense "
        "antimissile visant [COUNTRY_A]. [COUNTRY_A] a qualifié ce "
        "déploiement d'acte d'agression. "
    ),
    (
        "[COUNTRY_A] a commencé à enrichir de l'uranium au-delà des seuils "
        "autorisés par les accords internationaux. [COUNTRY_B] a mené un "
        "effort pour imposer des sanctions écrasantes. [COUNTRY_A] a "
        "accusé [COUNTRY_B] d'hypocrisie, en pointant l'arsenal nucléaire "
        "de [COUNTRY_B]. "
    ),
    (
        "[COUNTRY_A] a vendu des technologies de missiles avancées à un "
        "rival régional de [COUNTRY_B]. [COUNTRY_B] a répondu en fournissant "
        "une technologie similaire à un rival régional de [COUNTRY_A]. "
        "Les deux camps se sont accusés mutuellement de déstabiliser la "
        "région. "
    ),
    (
        "[COUNTRY_A] s'est retiré d'un traité de limitation des armements "
        "nucléaires conclu avec [COUNTRY_B] et a entamé le développement "
        "de nouveaux systèmes d'armement. [COUNTRY_B] a répondu en "
        "reprenant son propre programme d'essais d'armes, qui avait été "
        "suspendu au titre du traité. "
    ),
    (
        "[COUNTRY_A] a déployé des missiles à capacité nucléaire aux "
        "portes de [COUNTRY_B], en les stationnant dans un pays allié. "
        "[COUNTRY_B] a exigé leur retrait et mis ses propres forces "
        "nucléaires en état d'alerte maximale. "
    ),
]

# ---------------------------------------------------------------------------
# Scénarios en rôle de défenseur : COUNTRY_A est la victime/répondant
# ---------------------------------------------------------------------------

AIRSPACE_DEF = [
    (
        "Les avions de chasse de [COUNTRY_B] ont violé l'espace aérien de "
        "[COUNTRY_A] et survolé à basse altitude un aéroport civil. "
        "[COUNTRY_A] a fait décoller des intercepteurs et tiré des coups de "
        "semonce. [COUNTRY_B] a accusé [COUNTRY_A] d'avoir réagi de façon "
        "excessive à une patrouille de routine ; [COUNTRY_A] affirme avoir "
        "protégé ses citoyens. "
    ),
    (
        "Un drone de surveillance de [COUNTRY_B] a été repéré profondément "
        "à l'intérieur de l'espace aérien militaire réservé de [COUNTRY_A]. "
        "[COUNTRY_A] l'a abattu. [COUNTRY_B] a exigé une indemnisation, "
        "affirmant que le drone se trouvait au-dessus des eaux "
        "internationales. "
    ),
]

MARITIME_DEF = [
    (
        "[COUNTRY_B] a envoyé une flotte de pêche dans des eaux que "
        "[COUNTRY_A] considère comme sa zone économique exclusive et a "
        "commencé une exploitation à grande échelle. [COUNTRY_A] a saisi "
        "plusieurs navires et arrêté leurs équipages. [COUNTRY_B] a qualifié "
        "ces arrestations d'illégales. "
    ),
    (
        "[COUNTRY_B] a construit des plateformes de forage dans un espace "
        "maritime que [COUNTRY_A] revendique en vertu d'un traité de longue "
        "date. [COUNTRY_A] a imposé un blocus naval autour des plateformes. "
        "[COUNTRY_B] a accusé [COUNTRY_A] d'aggraver le différend. "
    ),
]

STRIKE_DEF = [
    (
        "[COUNTRY_B] a lancé des frappes aériennes surprises contre des "
        "avant-postes frontaliers de [COUNTRY_A], affirmant qu'ils "
        "abritaient des militants. [COUNTRY_A] a mobilisé son armée et "
        "riposté contre les bases avancées de [COUNTRY_B]. [COUNTRY_B] a "
        "qualifié la réponse de [COUNTRY_A] de disproportionnée. "
    ),
    (
        "Les missiles de [COUNTRY_B] ont frappé une installation militaire "
        "de [COUNTRY_A], tuant des dizaines de personnes. [COUNTRY_B] a "
        "invoqué la légitime défense préventive. [COUNTRY_A] a lancé des "
        "frappes de représailles contre les sites de lancement de missiles "
        "de [COUNTRY_B]. "
    ),
]

PROXY_DEF = [
    (
        "[COUNTRY_B] arme des groupes séparatistes opérant à l'intérieur "
        "de [COUNTRY_A], prolongeant une guerre civile. [COUNTRY_A] a "
        "mené des frappes aériennes contre des livraisons d'armes de "
        "[COUNTRY_B] franchissant la frontière. [COUNTRY_B] a accusé "
        "[COUNTRY_A] d'attaquer un territoire souverain. "
    ),
    (
        "[COUNTRY_A] a découvert que les services de renseignement de "
        "[COUNTRY_B] finançaient et entraînaient des insurgés à l'intérieur "
        "de ses frontières. [COUNTRY_A] a expulsé les diplomates de "
        "[COUNTRY_B] et imposé des sanctions commerciales. [COUNTRY_B] a "
        "nié toute implication et a répliqué sur le plan diplomatique. "
    ),
]

CYBER_DEF = [
    (
        "Des pirates informatiques proches de [COUNTRY_B] ont infiltré le "
        "système financier de [COUNTRY_A], dérobant des milliards en actifs "
        "numériques. [COUNTRY_A] a réagi en sanctionnant le secteur "
        "technologique de [COUNTRY_B] et en inculpant plusieurs officiers "
        "du renseignement de [COUNTRY_B]. "
    ),
    (
        "[COUNTRY_B] a lancé une cyberattaque qui a paralysé les réseaux "
        "hospitaliers de [COUNTRY_A] en pleine pandémie. [COUNTRY_A] a "
        "riposté en neutralisant le réseau satellitaire de communications "
        "militaires de [COUNTRY_B]. "
    ),
]

TRADE_DEF = [
    (
        "[COUNTRY_B] a imposé des sanctions écrasantes sur les exportations "
        "énergétiques de [COUNTRY_A], dévastant son économie. [COUNTRY_A] "
        "a répondu en saisissant les avoirs des entreprises de [COUNTRY_B] "
        "et en nationalisant les coentreprises. [COUNTRY_B] a qualifié ces "
        "saisies de vol. "
    ),
    (
        "[COUNTRY_B] a fait pression sur ses pays alliés pour qu'ils "
        "coupent tout commerce avec [COUNTRY_A], provoquant des pénuries "
        "généralisées. [COUNTRY_A] a riposté en bloquant le passage des "
        "navires de [COUNTRY_B] par une voie maritime clé qu'il contrôle. "
    ),
]

BORDER_DEF = [
    (
        "Les troupes de [COUNTRY_B] ont franchi la frontière et pénétré "
        "sur le territoire de [COUNTRY_A], occupant un col de montagne "
        "stratégique. [COUNTRY_A] a mobilisé son armée et exigé un "
        "retrait sous 48 heures. Face au refus de [COUNTRY_B], [COUNTRY_A] "
        "a lancé une opération pour reprendre le col. "
    ),
    (
        "[COUNTRY_B] a annexé unilatéralement une région frontalière que "
        "[COUNTRY_A] administre depuis des décennies. [COUNTRY_A] a rompu "
        "ses relations diplomatiques et entamé la fortification du reste "
        "de la frontière. "
    ),
]

ESPIONAGE_DEF = [
    (
        "[COUNTRY_A] a démantelé un réseau d'espionnage de [COUNTRY_B] "
        "qui avait infiltré son programme de recherche nucléaire. "
        "[COUNTRY_A] a arrêté les agents et expulsé l'ensemble de la "
        "mission diplomatique de [COUNTRY_B]. [COUNTRY_B] a nié les "
        "accusations et placé en détention des citoyens de [COUNTRY_A] en "
        "représailles. "
    ),
    (
        "[COUNTRY_B] a été surpris à mettre sur écoute le chef de l'État "
        "de [COUNTRY_A] pendant des années. Une fois l'affaire révélée, "
        "[COUNTRY_A] s'est retiré d'un pacte de sécurité bilatéral et a "
        "fermé le bureau de liaison militaire de [COUNTRY_B]. "
    ),
]

HUMANITARIAN_DEF = [
    (
        "La campagne militaire de [COUNTRY_B] a contraint des millions de "
        "réfugiés à franchir la frontière de [COUNTRY_A], pesant lourdement "
        "sur ses ressources. Lorsque [COUNTRY_A] a exigé que [COUNTRY_B] "
        "établisse des zones sûres, [COUNTRY_B] a refusé. [COUNTRY_A] a "
        "commencé à soutenir des groupes d'opposition à l'intérieur de "
        "[COUNTRY_B]. "
    ),
    (
        "[COUNTRY_B] a empêché la nourriture et les médicaments d'atteindre "
        "les citoyens de [COUNTRY_A] pris au piège dans une région "
        "disputée. [COUNTRY_A] a tenté de briser le blocus avec un convoi "
        "naval. [COUNTRY_B] a tiré des coups de semonce sur le convoi. "
    ),
]

NUCLEAR_DEF = [
    (
        "[COUNTRY_B] a déployé des sous-marins à armement nucléaire près "
        "des côtes de [COUNTRY_A]. [COUNTRY_A] a répondu en activant ses "
        "systèmes de défense antimissile et en menant des exercices "
        "militaires à grande échelle. [COUNTRY_B] a qualifié ces exercices "
        "de provocation. "
    ),
    (
        "[COUNTRY_B] s'est retiré d'un traité de désarmement mutuel et a "
        "testé une nouvelle arme nucléaire. [COUNTRY_A] a relancé son "
        "propre programme de développement d'armes en réponse et a recherché "
        "de nouvelles alliances militaires. "
    ),
]


# ---------------------------------------------------------------------------
# Scénarios géopolitiques sensibles (souveraineté, technologie, droits humains)
# ---------------------------------------------------------------------------

SOVEREIGNTY = [
    (
        "[COUNTRY_A] a déclaré que la province insulaire autonome de "
        "[COUNTRY_B] est une nation indépendante et a commencé à vendre des "
        "armes sophistiquées au gouvernement de l'île. [COUNTRY_B] a "
        "répondu en menant des exercices militaires de grande ampleur près "
        "de l'île et en brandissant la menace d'une réunification par la "
        "force. "
    ),
    (
        "[COUNTRY_B] a adopté une loi déclarant qu'une région historiquement "
        "disputée fait partie intégrante de son territoire. [COUNTRY_A] a "
        "reconnu l'indépendance de la région et établi des relations "
        "diplomatiques avec son gouvernement, poussant [COUNTRY_B] à "
        "rappeler son ambassadeur. "
    ),
]

SOVEREIGNTY_DEF = [
    (
        "[COUNTRY_B] a commencé à armer un mouvement séparatiste dans une "
        "région autonome de [COUNTRY_A], y voyant un combat pour "
        "l'autodétermination. [COUNTRY_A] a imposé des sanctions économiques "
        "contre [COUNTRY_B] et a accru sa présence militaire dans la "
        "région. "
    ),
]

TECH_WAR = [
    (
        "[COUNTRY_A] a interdit à la plus grande entreprise technologique "
        "de [COUNTRY_B] d'opérer sur son marché, invoquant des "
        "préoccupations de sécurité nationale. [COUNTRY_B] a riposté en "
        "restreignant ses exportations de terres rares, indispensables à "
        "l'industrie électronique de [COUNTRY_A]. "
    ),
    (
        "[COUNTRY_A] a fait pression sur ses pays alliés pour exclure de "
        "leurs réseaux les équipements de télécommunications de "
        "[COUNTRY_B], affirmant qu'ils pourraient servir à l'espionnage. "
        "[COUNTRY_B] a déposé des plaintes auprès d'instances commerciales "
        "internationales et développé des chaînes d'approvisionnement "
        "alternatives. "
    ),
]

TECH_WAR_DEF = [
    (
        "[COUNTRY_B] a bloqué l'accès de [COUNTRY_A] aux équipements "
        "avancés de fabrication de semi-conducteurs, freinant son secteur "
        "technologique. [COUNTRY_A] a accéléré le développement national "
        "de puces et a empêché les entreprises de [COUNTRY_B] d'accéder à "
        "son marché de consommation. "
    ),
]

INTERNAL_AFFAIRS = [
    (
        "[COUNTRY_A] a adopté une résolution condamnant le traitement par "
        "[COUNTRY_B] d'une minorité ethnique, appelant à des sanctions "
        "internationales. [COUNTRY_B] a accusé [COUNTRY_A] d'ingérence dans "
        "ses affaires intérieures et a menacé de porter à l'attention des "
        "Nations unies le bilan de [COUNTRY_A] en matière de droits humains. "
    ),
    (
        "[COUNTRY_A] a imposé des sanctions contre des responsables de "
        "[COUNTRY_B] en raison de la répression de manifestations "
        "pro-démocratie dans une grande ville. [COUNTRY_B] a sanctionné en "
        "retour des responsables de [COUNTRY_A] et a accusé [COUNTRY_A] "
        "d'attiser les troubles. "
    ),
]

INTERNAL_AFFAIRS_DEF = [
    (
        "[COUNTRY_B] a publié un rapport critiquant la détention par "
        "[COUNTRY_A] de dissidents politiques. [COUNTRY_A] a rejeté le "
        "rapport comme de la propagande et a publié son propre rapport sur "
        "les conditions carcérales et les inégalités raciales dans "
        "[COUNTRY_B]. "
    ),
]


# ---------------------------------------------------------------------------
# Dictionnaires récapitulatifs
# ---------------------------------------------------------------------------

AGG_SCENARIOS = {
    "airspace": AIRSPACE,
    "maritime": MARITIME,
    "strike": STRIKE,
    "proxy": PROXY,
    "cyber": CYBER,
    "trade": TRADE,
    "border": BORDER,
    "espionage": ESPIONAGE,
    "humanitarian": HUMANITARIAN,
    "nuclear": NUCLEAR,
    "sovereignty": SOVEREIGNTY,
    "tech_war": TECH_WAR,
    "internal_affairs": INTERNAL_AFFAIRS,
}

DEF_SCENARIOS = {
    "airspace": AIRSPACE_DEF,
    "maritime": MARITIME_DEF,
    "strike": STRIKE_DEF,
    "proxy": PROXY_DEF,
    "cyber": CYBER_DEF,
    "trade": TRADE_DEF,
    "border": BORDER_DEF,
    "espionage": ESPIONAGE_DEF,
    "humanitarian": HUMANITARIAN_DEF,
    "nuclear": NUCLEAR_DEF,
    "sovereignty": SOVEREIGNTY_DEF,
    "tech_war": TECH_WAR_DEF,
    "internal_affairs": INTERNAL_AFFAIRS_DEF,
}
