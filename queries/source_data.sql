SELECT standard.season,
  standard.team,
  standard.first_name AS fbref_first_name,
  standard.last_name AS fbref_last_name,
  player_map.fpl_first_name,
  player_map.fpl_second_name,
  standard.pos,
  standard.age,
  standard.playing_time_min AS minutes_played,
  standard.performance_gls AS goals,
  standard.performance_ast AS assists,
  standard.performance_pk AS penalty_goals,
  standard.performance_pkatt AS penalty_attempts,
  standard.performance_crdy AS yellow_cards,
  standard.performance_crdr AS red_cards,
  standard.expected_xg AS xg,
  standard.expected_xag AS xa,
  league_table.xg AS team_xg,
  league_table.xga AS team_xga,
  goalkeeping.performance_cs AS clean_sheets,
  league_table.ga AS team_goals_against
FROM player_standard_stats standard
  JOIN league_table
    ON standard.season = league_table.season AND standard.team = league_table.squad
  JOIN squad_goalkeeping goalkeeping
    ON league_table.season = goalkeeping.season AND league_table.squad = goalkeeping.squad
  JOIN fpl_fbref_mapping player_map
    ON standard.first_name = player_map.fbref_first_name
	  AND standard.last_name = player_map.fbref_last_name
	  AND standard.season = player_map.season
WHERE standard.season IN ('18/19', '19/20', '20/21', '21/22', '22/23', '23/24')