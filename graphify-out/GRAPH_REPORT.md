# Graph Report - HawkEye-shubh-handoff  (2026-10-22)

## Corpus Check
- 105 files · ~65,690 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: (none) 3, .pt 2)

## Summary
- 1150 nodes · 2947 edges · 71 communities (50 shown, 21 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 247 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b3ca0c27`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pipeline.py
- react-native
- PersonDetection
- train_action.py
- test_action_pipeline.py
- evaluate.py
- StageA
- [id].tsx
- settings.tsx
- dependencies
- incidents.tsx
- expo
- types.ts
- test_pipeline_shutdown.py
- Pipeline
- ActionClassifier
- numpy
- numpy
- train_weapon.py
- package.json
- IdentityService
- app.py
- drone.tsx
- Modules
- 0. Set up your environment
- people.tsx
- pathlib
- attendance.tsx
- WeaponDetector
- StreamHub
- action_model.py
- (tabs)/_layout.tsx
- HawkEye — Indoor Surveillance & Attendance
- pipeline.py
- HawkEye: Complete Architecture & Technical Deep Dive
- PoseDetector
- .__init__
- app/_layout.tsx
- Training Guide — improving the models
- Pipeline
- json
- augment
- client.ts
- config.py
- HawkEye App — Context Snapshot
- scripts
- settings.ts
- tsconfig.json
- 4. Deep Dive into System Subsystems
- queries.ts
- ActionNet
- _index_response
- ScreenSource
- HawkEye — Ethics, Privacy & Legal (India)
- HawkEye — Setup
- dispatch.py
- metro.config.js
- VideoSource
- config.py
- train_all.sh
- delete
- group_split
- extract_clip_features
- StubStore
- box_iou
- pick_thresholds

## God Nodes (most connected - your core abstractions)
1. `PersonDetection` - 35 edges
2. `StageAEvent` - 28 edges
3. `react-native` - 27 edges
4. `Pipeline` - 27 edges
5. `react` - 26 edges
6. `frame_feature()` - 26 edges
7. `Reasoning` - 26 edges
8. `Palette` - 25 edges
9. `StageA` - 25 edges
10. `AnomalyType` - 25 edges

## Surprising Connections (you probably didn't know these)
- `6. Where each piece lives (quick map)` --references--> `ActionNet`  [INFERRED]
  docs/training.md → hawkeye/detect/action_model.py
- `Solving the Violence vs. Fall Confusion` --references--> `ActionNet`  [INFERRED]
  HAWKEYE_DEEP_DIVE.md → hawkeye/detect/action_model.py
- `Roadmap` --references--> `DroneSource`  [INFERRED]
  README.md → hawkeye/ingest/sources.py
- `5. Train the weapon detector (knife / gun / etc.)` --references--> `convert_voc_to_yolo()`  [INFERRED]
  docs/training.md → scripts/train_weapon.py
- `Continuous Passive Attendance (`hawkeye/attendance/`)` --references--> `attendance()`  [INFERRED]
  HAWKEYE_DEEP_DIVE.md → server/app.py

## Import Cycles
- 3-file cycle: `hawkeye/detect/__init__.py -> hawkeye/detect/action_classifier.py -> hawkeye/detect/skeleton_features.py -> hawkeye/detect/__init__.py`
- 3-file cycle: `hawkeye/detect/__init__.py -> hawkeye/detect/action_classifier.py -> hawkeye/detect/heuristics.py -> hawkeye/detect/__init__.py`

## Communities (71 total, 21 thin omitted)

### Community 0 - "pipeline.py"
Cohesion: 0.06
Nodes (34): What exists in the backend (unmodified, for reference), DecisionConfig, DecisionEngine, ActionRecord, ActionType, AnomalyType, BBox, FaceMatch (+26 more)

### Community 1 - "react-native"
Cohesion: 0.11
Nodes (34): ICONS, styles, CONNECTION_META, Props, styles, Props, styles, Variant (+26 more)

### Community 2 - "PersonDetection"
Cohesion: 0.14
Nodes (11): aggressive_pose(), leg_horizontalness(), proximity(), analyze(), fall_score(), is_human(), is_standing(), torso_points() (+3 more)

### Community 3 - "train_action.py"
Cohesion: 0.11
Nodes (14): standardize(), build_dataset(), cap_per_class(), confusion(), cross_confusion_rate(), extract_clip(), flush(), fit_temperature() (+6 more)

### Community 4 - "test_action_pipeline.py"
Cohesion: 0.12
Nodes (19): build_model(), build_sequence(), WeaponDetection, calm_pair_sequence(), fall_sequence(), main(), make_person(), test_calm_pair_is_not_violent_looking() (+11 more)

### Community 5 - "evaluate.py"
Cohesion: 0.16
Nodes (9): bench_fps(), compare_methods(), confusion(), iter_labelled_clips(), LoadedModel, main(), print_report(), test_action() (+1 more)

### Community 7 - "[id].tsx"
Cohesion: 0.12
Nodes (35): FaceRow(), Hero(), IconName, Identity(), IncidentDetailScreen(), joinList(), Row(), SectionCard() (+27 more)

### Community 8 - "settings.tsx"
Cohesion: 0.15
Nodes (14): CONNECTION_META, HEALTH_LABELS, healthLabel(), HealthRow(), SettingsCard(), SettingsScreen(), styles, useHealth() (+6 more)

### Community 9 - "dependencies"
Cohesion: 0.07
Nodes (27): dependencies, expo, expo-asset, expo-blur, expo-camera, expo-constants, expo-image-picker, expo-linear-gradient (+19 more)

### Community 10 - "incidents.tsx"
Cohesion: 0.15
Nodes (25): DroneScreen(), dotColorFor(), FilterChip(), FilterKey, FILTERS, IncidentsScreen(), styles, FeedPill() (+17 more)

### Community 11 - "expo"
Cohesion: 0.08
Nodes (24): package, typedRoutes, expo, android, backgroundColor, experiments, ios, name (+16 more)

### Community 12 - "types.ts"
Cohesion: 0.13
Nodes (14): ActionRecord, AnomalyType, BBox, FaceMatch, Health, IncidentStatus, Person, Reasoning (+6 more)

### Community 13 - "test_pipeline_shutdown.py"
Cohesion: 0.18
Nodes (9): _now(), main(), make_pipeline(), StubHub, test_join_happens_before_close(), worker(), test_slow_thread_does_not_hang_shutdown(), test_stop_is_idempotent_and_restartable() (+1 more)

### Community 15 - "ActionClassifier"
Cohesion: 0.17
Nodes (3): ActionClassifier, label_to_type(), torso_horizontalness()

### Community 16 - "numpy"
Cohesion: 0.20
Nodes (10): _actor_frame(), _body_spread(), fit_standardizer(), frame_feature(), _kp(), _mid(), _min_pair_distance(), _norm_kps() (+2 more)

### Community 18 - "train_weapon.py"
Cohesion: 0.05
Nodes (28): _cpu_count(), get_profile(), get_yolo_kwargs(), TrainProfile, authenticate(), check_kaggle_json(), download(), _print_token_instructions() (+20 more)

### Community 19 - "package.json"
Cohesion: 0.10
Nodes (20): devDependencies, @types/react, typescript, main, name, private, version, expo-asset (+12 more)

### Community 20 - "IdentityService"
Cohesion: 0.08
Nodes (6): Person, Face, IdentityRecognizer, IdentityService, EmbeddingStore, main()

### Community 21 - "app.py"
Cohesion: 0.14
Nodes (5): enroll(), incidents(), people(), set_incident_status(), ws()

### Community 22 - "drone.tsx"
Cohesion: 0.18
Nodes (12): HeaderBar(), DPadButton(), enter(), IconName, PatrolRoute, ROUTES, styles, Telemetry (+4 more)

### Community 23 - "Modules"
Cohesion: 0.22
Nodes (7): Data flow, Design choices, Extension points, HawkEye — Architecture, Modules, PoseDetector, MotionAnomaly

### Community 24 - "0. Set up your environment"
Cohesion: 0.25
Nodes (8): 0.1 Get the repo, 0.2 Set up the Python environment, 0.2b Run the self-checks first, 0.3 Set up the Kaggle API token (for automatic dataset download), 0.4 Get the datasets, 0.5 Train, then commit + push your CODE changes, 0.6 Training on Windows + NVIDIA CUDA, 0. Set up your environment

### Community 25 - "people.tsx"
Cohesion: 0.21
Nodes (15): errorMessage(), initials(), LabeledInput(), metaLine(), PeopleScreen(), PersonRow(), PersonSkeleton(), styles (+7 more)

### Community 26 - "pathlib"
Cohesion: 0.17
Nodes (3): AttendanceStore, date_str(), AttendanceRecord

### Community 27 - "attendance.tsx"
Cohesion: 0.30
Nodes (14): AttendanceScreen(), AttendanceSkeletonRow(), fullDateLabel(), Hero(), parseIsoDate(), relativeDayLabel(), shiftDate(), StepperButton() (+6 more)

### Community 28 - "WeaponDetector"
Cohesion: 0.27
Nodes (3): WeaponDetector, test_weapon_detector_degrades_on_missing_file(), test_weapon_label_matching()

### Community 31 - "(tabs)/_layout.tsx"
Cohesion: 0.67
Nodes (3): TabsLayout(), AnimatedTabBar(), TabItem()

### Community 32 - "HawkEye — Indoor Surveillance & Attendance"
Cohesion: 0.10
Nodes (17): A physical phone on the same Wi-Fi, HawkEye — Ops App, Iterating on the UI, One address (the normal way), Running, Screens, Architecture, Configure (+9 more)

### Community 34 - "HawkEye: Complete Architecture & Technical Deep Dive"
Cohesion: 0.20
Nodes (9): 1. Executive Summary: What is HawkEye?, 2. Core Architecture: The Fast/Slow Cascade & Multi-Thread Pipeline, 5. Model Training, Datasets & Benchmarking (`scripts/`, `eval/`), 7. Project Directory Structure Map, 8. Summary of Unique Selling Points (USPs), HawkEye: Complete Architecture & Technical Deep Dive, Multi-Threading Design, The Dual-Mission Model (+1 more)

### Community 37 - "app/_layout.tsx"
Cohesion: 0.17
Nodes (15): AppShell(), queryClient, RootLayout(), styles, AttendanceRecord, WsEvent, ToastHost(), ToastRow() (+7 more)

### Community 38 - "Training Guide — improving the models"
Cohesion: 0.15
Nodes (13): 2. Get the datasets, 2a. Automated (Kaggle) — one command, 2b. Manual (non-Kaggle sources) — optional accuracy boost, 3. Train the action classifier, 4.1 Re-tune the config from the run's metrics, 4. Plug the trained model in (no code changes), 5. Train the weapon detector (knife / gun / etc.), 6. Where each piece lives (quick map) (+5 more)

### Community 40 - "json"
Cohesion: 0.15
Nodes (11): Architecture, 1. Vision, Machine Learning & Neural Networks, 2. Backend & Server Infrastructure, 3. Frontend & Mobile Operator App (`app/`), 3. Technology Stack: Every Tech Used & What It's For, attendance(), evidence(), health() (+3 more)

### Community 41 - "augment"
Cohesion: 0.28
Nodes (7): augment(), drop_actor_b(), flip_sequence(), _kp_blocks(), rotate_sequence(), scale_sequence(), test_augmentation_is_consistent()

### Community 42 - "client.ts"
Cohesion: 0.43
Nodes (5): ApiError, apiUrl(), base(), request(), wsUrl()

### Community 43 - "config.py"
Cohesion: 0.22
Nodes (7): load_config(), main(), get_config(), get_hub(), set_config(), setup_logging(), lifespan()

### Community 44 - "HawkEye App — Context Snapshot"
Cohesion: 0.25
Nodes (7): App architecture, Design decisions (locked, from `/model` plan-mode session), HawkEye App — Context Snapshot, How to run, Key conventions to preserve when extending, Not done yet (future phases, per the plan file), Status: Phase 1 complete

### Community 45 - "scripts"
Cohesion: 0.25
Nodes (8): scripts, android, build:web, ios, lint, start, typecheck, web

### Community 46 - "settings.ts"
Cohesion: 0.31
Nodes (7): DEFAULT_BASE, load(), normalize(), persist(), SettingsState, useSettings, expo-secure-store

### Community 47 - "tsconfig.json"
Cohesion: 0.25
Nodes (7): compilerOptions, baseUrl, paths, strict, extends, include, expo/tsconfig.base

### Community 48 - "4. Deep Dive into System Subsystems"
Cohesion: 0.18
Nodes (11): 4.2 Stage A: Real-Time Detection & Kinematics (`hawkeye/detect/`), 4.3 Stage B: Vision-Language Model Reasoning (`hawkeye/reason/`), 4.4 Biometric Identity & Passive Attendance (`hawkeye/identity/` & `hawkeye/attendance/`), 4.5 Gated Decision Engine & Dispatch (`hawkeye/decision/`), 4.6 Server, Streaming & WebSocket Protocol (`server/app.py`), 4.7 Frontend Companion Console (`app/`), 4. Deep Dive into System Subsystems, Biometric Engine (`hawkeye/identity/`) (+3 more)

### Community 49 - "queries.ts"
Cohesion: 0.17
Nodes (11): api, AttendanceList, AttendanceT, DeleteResult, HealthT, IncidentList, INCIDENTS_LIMIT, IncidentT (+3 more)

### Community 51 - "_index_response"
Cohesion: 0.33
Nodes (3): index(), _index_response(), spa_fallback()

### Community 53 - "HawkEye — Ethics, Privacy & Legal (India)"
Cohesion: 0.40
Nodes (4): HawkEye — Ethics, Privacy & Legal (India), Known limitations to state honestly (thesis), Phase 3 commitments (production), What we do

### Community 54 - "HawkEye — Setup"
Cohesion: 0.40
Nodes (4): Camera options, HawkEye — Setup, macOS (development + training), Reasoner backends

### Community 55 - "dispatch.py"
Cohesion: 0.27
Nodes (4): build_dispatchers(), Dispatcher, LogDispatcher, WebhookDispatcher

### Community 56 - "metro.config.js"
Cohesion: 0.50
Nodes (3): config, { getDefaultConfig }, expo

### Community 57 - "VideoSource"
Cohesion: 0.24
Nodes (3): 4.1 Ingest Layer (`hawkeye/ingest/`), DroneSource, VideoSource

### Community 58 - "config.py"
Cohesion: 0.31
Nodes (6): DetectConfig, IdentityConfig, IngestConfig, ReasonConfig, ServerConfig, StorageConfig

### Community 61 - "delete"
Cohesion: 0.33
Nodes (3): clear_incidents(), delete_person(), delete_single_incident()

### Community 66 - "group_split"
Cohesion: 0.22
Nodes (6): 1. Why violence and fall got confused (and the fix), 1a. The two bugs that made a "0.95 F1" model behave randomly, 1b. The modelling problem, group_split(), window_index(), test_group_split_has_no_leakage()

## Knowledge Gaps
- **13 isolated node(s):** `@expo/metro-runtime`, `expo-asset`, `expo-camera`, `expo-constants`, `expo-linking` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 483 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HawkEye App — Context Snapshot` connect `HawkEye App — Context Snapshot` to `pipeline.py`, `[id].tsx`?**
  _High betweenness centrality (0.396) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `PersonDetection` (e.g. with `aggressive_pose()` and `leg_horizontalness()`) actually correct?**
  _`PersonDetection` has 18 INFERRED edges - model-reasoned connections that need verification._
- **What connects `@expo/metro-runtime`, `expo-asset`, `expo-camera` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `pipeline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06398046398046398 - nodes in this community are weakly interconnected._
- **Why does `What exists in the backend (unmodified, for reference)` connect `pipeline.py` to `HawkEye App — Context Snapshot`?**
  _High betweenness centrality (0.395) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `StageAEvent` (e.g. with `What exists in the backend (unmodified, for reference)` and `4.2 Stage A: Real-Time Detection & Kinematics (`hawkeye/detect/`)`) actually correct?**
  _`StageAEvent` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Should `react-native` be split into smaller, more focused modules?**
  _Cohesion score 0.10558069381598793 - nodes in this community are weakly interconnected._