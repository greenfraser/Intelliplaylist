#defined banned_artist/1.
#defined required_artist/1.
#defined artist_match/2.
#defined artist_target_count/1.

#defined banned_album/1.
#defined required_album/1.
#defined album_match/2.

#defined banned_track/1.
#defined required_track/1.
#defined track_match/2.

#defined max_tracks_per_artist/1.

{ in_playlist(ID) : song(ID, _, _, _, _, _, _, _, _, _, _, _) }.
:- #count { ID : in_playlist(ID) } != n.

:- banned_artist(A), in_playlist(ID), artist_match(A, ID).
:- required_artist(A), 0 = #count { ID : in_playlist(ID), artist_match(A, ID) }.

artist_requested_track(ID) :-
    required_artist(A),
    artist_match(A, ID).

:- artist_target_count(M),
   #count { ID : in_playlist(ID), artist_requested_track(ID) } < M.

:- banned_album(A), in_playlist(ID), album_match(A, ID).
:- required_album(A), 0 = #count { ID : in_playlist(ID), album_match(A, ID) }.

:- banned_track(T), in_playlist(ID), track_match(T, ID).
:- required_track(T), 0 = #count { ID : in_playlist(ID), track_match(T, ID) }.

artist_track_count(A, N) :-
    performer(A, _),
    N = #count { ID : in_playlist(ID), performer(A, ID) }.

:- max_tracks_per_artist(M),
   artist_track_count(A, N),
   N > M.

protected_candidate(ID) :-
    required_artist(A),
    artist_match(A, ID).

protected_candidate(ID) :-
    required_album(A),
    album_match(A, ID).

protected_candidate(ID) :-
    required_track(T),
    track_match(T, ID).

:- required_track(T), 0 = #count { ID : in_playlist(ID), track_match(T, ID) }.

repeat_artist_penalty(A, P) :-
    artist_track_count(A, N),
    N > 1,
    P = N - 1.

valence_penalty(ID, D) :-
    in_playlist(ID),
    not protected_candidate(ID),
    song(ID, _, _, V, _, _, _, _, _, _, _, _),
    target_valence(T),
    V >= T,
    D = V - T.
valence_penalty(ID, D) :-
    in_playlist(ID),
    not protected_candidate(ID),
    song(ID, _, _, V, _, _, _, _, _, _, _, _),
    target_valence(T),
    V < T,
    D = T - V.

energy_penalty(ID, D) :-
    in_playlist(ID),
    not protected_candidate(ID),
    song(ID, _, E, _, _, _, _, _, _, _, _, _),
    target_energy(T),
    E >= T,
    D = E - T.
energy_penalty(ID, D) :-
    in_playlist(ID),
    not protected_candidate(ID),
    song(ID, _, E, _, _, _, _, _, _, _, _, _),
    target_energy(T),
    E < T,
    D = T - E.

tempo_penalty(ID, D) :-
    in_playlist(ID),
    not protected_candidate(ID),
    song(ID, TM, _, _, _, _, _, _, _, _, _, _),
    target_tempo(T),
    TM >= T,
    D = TM - T.
tempo_penalty(ID, D) :-
    in_playlist(ID),
    not protected_candidate(ID),
    song(ID, TM, _, _, _, _, _, _, _, _, _, _),
    target_tempo(T),
    TM < T,
    D = T - TM.

#minimize { P@4, A : repeat_artist_penalty(A, P) }.
#minimize { D@3, ID : valence_penalty(ID, D) }.
#minimize { D@2, ID : energy_penalty(ID, D) }.
#minimize { D@1, ID : tempo_penalty(ID, D) }.

#show in_playlist/1.
