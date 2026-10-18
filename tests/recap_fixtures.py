"""A frozen recap case for the recap verifier tests: the FT1 example as it stood before the prompt examples changed
(an equaliser, three scorer claims, a star who did not score). The live examples (prompts.RECAP_EXAMPLES) are
tested on their own."""
from xplain.prompts import RecapExample, _star, _team
from xplain.recap import RecapCard, RecapGoal, RecapOut

_HBA, _KWR = "Harbourne Athletic", "Kellsworth Rovers"

FT = RecapExample(
    name="FT fixture (comeback, late winner, star who did not score)",
    card=RecapCard(
        scope="full_time", match_id="example", home=_team("hba", _HBA, "above"),
        away=_team("kwr", _KWR, "below"), score_home=2, score_away=1,
        goals=[RecapGoal(team_id="kwr", team_name=_KWR, player_name="Dario Pell", minute_label="31",
                         period=1, late=False, home=0, away=1),
               RecapGoal(team_id="hba", team_name=_HBA, player_name="Joss Marlin",
                         minute_label="62", period=2, late=False, home=1, away=1),
               RecapGoal(team_id="hba", team_name=_HBA, player_name="Tomas Avery",
                         minute_label="90+2", period=2, late=True, home=2, away=1)],
        result="home_win", comeback_team_id="hba", late_winner_team_id="hba", late_winner_scorer="Tomas Avery",
        star=_star("Ollie Branck", "hba", _HBA, 6)),
    out=RecapOut(
        text_en="Harbourne Athletic came from behind to beat Kellsworth Rovers, with Tomas Avery's late winner "
                "after Joss Marlin had cancelled out Dario Pell's opener. Harbourne Athletic completed "
                "{home_vs_expected} more passes than expected; Kellsworth Rovers completed {away_vs_expected} "
                "fewer than expected. {star_name} stood out with {star_pva} more passes than expected.",
        text_es="Harbourne Athletic remontó y venció a Kellsworth Rovers con un gol tardío de Tomas Avery, después "
                "de que Joss Marlin igualara el tanto inicial de Dario Pell. Harbourne Athletic completó "
                "{home_vs_expected} pases más de lo esperado; Kellsworth Rovers completó {away_vs_expected} menos "
                "de lo esperado. {star_name} destacó con {star_pva} pases más de lo esperado.",
        text_el="Η Harbourne Athletic έκανε την ανατροπή και νίκησε την Kellsworth Rovers με γκολ του Tomas Avery "
                "στο φινάλε, μετά την ισοφάριση του Joss Marlin στο γκολ του Dario Pell. Η Harbourne Athletic "
                "ολοκλήρωσε {home_vs_expected} περισσότερες πάσες από τις αναμενόμενες· η Kellsworth Rovers "
                "{away_vs_expected} λιγότερες από τις αναμενόμενες. Ο {star_name} ξεχώρισε με {star_pva} "
                "περισσότερες πάσες από τις αναμενόμενες.",
        claims_used=["result", "comeback:Harbourne Athletic", "late_winner:Harbourne Athletic",
                     "scorer:Dario Pell", "scorer:Joss Marlin", "scorer:Tomas Avery",
                     "passing:Harbourne Athletic", "passing:Kellsworth Rovers", "star:Ollie Branck"]),
)
