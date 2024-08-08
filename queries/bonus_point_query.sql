SELECT pass.first_name,
  pass.last_name,
  pass.season,
  pass.team,
  crspa AS crosses,
  sca_types_passlive + sca_types_passdead AS key_passes,
  blocks_blocks + def.interceptions + def.clr AS cba,
  COALESCE(tackles_tklw * 1.0 / NULLIF(tackles_tkl, 0), 0) AS tkl_pct,
  pos.take_minus_ons_succ AS dribbles,
  mis.performance_fld AS fouled,
  shoot.standard_sot AS shots_on_target,
  pass.total_cmp_pct AS pass_completion,
  mis.performance_off AS offsides,
  mis.performance_fls AS fouls,
  pos.take_minus_ons_tkld AS tackled
FROM player_passing pass
JOIN player_goal_and_shot_creation shot
  ON pass.first_name = shot.first_name
    AND pass.last_name = shot.last_name
	AND pass.season = shot.season
	AND pass.team = shot.team
JOIN player_defensive_actions def
  ON pass.first_name = def.first_name
    AND pass.last_name = def.last_name
	AND pass.season = def.season
	AND pass.team = def.team
JOIN player_possession pos
  ON pass.first_name = pos.first_name
    AND pass.last_name = pos.last_name
	AND pass.season = pos.season
	AND pass.team = pos.team
JOIN player_miscellaneous_stats mis
  ON pass.first_name = mis.first_name
    AND pass.last_name = mis.last_name
	AND pass.season = mis.season
	AND pass.team = mis.team
JOIN player_shooting shoot
  ON pass.first_name = shoot.first_name
    AND pass.last_name = shoot.last_name
	AND pass.season = shoot.season
	AND pass.team = shoot.team
