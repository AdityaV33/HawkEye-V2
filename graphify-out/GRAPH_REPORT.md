# Graph Report - HawkEye-shubh-handoff  (2026-10-30)

## Corpus Check
- 107 files · ~75,694 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: .pt 4, (none) 3)

## Summary
- 1191 nodes · 3095 edges · 63 communities (51 shown, 12 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 281 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9ca21d33`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- StageAEvent
- react-native
- StageA
- test_action_pipeline.py
- test_action_pipeline.py
- evaluate.py
- people.tsx
- [id].tsx
- train_weapon.py
- dependencies
- incidents.tsx
- expo
- settings.tsx
- test_pipeline_shutdown.py
- IncidentStore
- ActionClassifier
- numpy
- numpy
- drone.tsx
- package.json
- get
- app.py
- queries.ts
- WeaponDetector
- Pipeline
- config.py
- attendance/store.py
- attendance.tsx
- test_fire_detector.py
- StreamHub
- action_model.py
- types.ts
- HawkEye — Indoor Surveillance & Attendance
- client.ts
- HawkEye: Complete Architecture & Technical Deep Dive
- config.py
- ._retention_loop
- app/_layout.tsx
- Training Guide — improving the models
- settings.ts
- typing
- HawkEye — Ops App
- _index_response
- runtime.py
- HawkEye App — Context Snapshot
- scripts
- delete
- tsconfig.json
- 4. Deep Dive into System Subsystems
- StubStore
- ActionNet
- 2. Get the datasets
- (tabs)/_layout.tsx
- HawkEye — Ethics, Privacy & Legal (India)
- HawkEye — Setup
- metro.config.js
- train_all.sh
- 3. Train the action classifier

## God Nodes (most connected - your core abstractions)
1. `PersonDetection` - 35 edges
2. `StageAEvent` - 32 edges
3. `AnomalyType` - 31 edges
4. `StageA` - 30 edges
5. `react-native` - 27 edges
6. `Pipeline` - 27 edges
7. `react` - 26 edges
8. `frame_feature()` - 26 edges
9. `Reasoning` - 26 edges
10. `Palette` - 25 edges

## Surprising Connections (you probably didn't know these)
- `6. Where each piece lives (quick map)` --references--> `ActionNet`  [INFERRED]
  docs/training.md → hawkeye/detect/action_model.py
- `Solving the Violence vs. Fall Confusion` --references--> `ActionNet`  [INFERRED]
  HAWKEYE_DEEP_DIVE.md → hawkeye/detect/action_model.py
- `5. Train the weapon detector (knife / gun / etc.)` --references--> `convert_voc_to_yolo()`  [INFERRED]
  docs/training.md → scripts/train_weapon.py
- `Continuous Passive Attendance (`hawkeye/attendance/`)` --references--> `attendance()`  [INFERRED]
  HAWKEYE_DEEP_DIVE.md → server/app.py
- `Modules` --references--> `AttendanceStore`  [INFERRED]
  docs/architecture.md → hawkeye/attendance/store.py

## Import Cycles
- 3-file cycle: `hawkeye/detect/__init__.py -> hawkeye/detect/action_classifier.py -> hawkeye/detect/skeleton_features.py -> hawkeye/detect/__init__.py`
- 3-file cycle: `hawkeye/detect/__init__.py -> hawkeye/detect/action_classifier.py -> hawkeye/detect/heuristics.py -> hawkeye/detect/__init__.py`

## Communities (63 total, 12 thin omitted)

### Community 0 - "StageAEvent"
Cohesion: 0.05
Nodes (35): What exists in the backend (unmodified, for reference), DecisionConfig, build_dispatchers(), Dispatcher, LogDispatcher, WebhookDispatcher, DecisionEngine, ActionRecord (+27 more)

### Community 1 - "react-native"
Cohesion: 0.10
Nodes (38): ICONS, styles, CONNECTION_META, Props, styles, Props, styles, Variant (+30 more)

### Community 2 - "StageA"
Cohesion: 0.17
Nodes (8): Extension points, 1a. The two bugs that made a "0.95 F1" model behave randomly, DetectConfig, FireDetection, MotionAnomaly, StageA, BBox, test_stage_a_fire_event_and_drawing()

### Community 3 - "test_action_pipeline.py"
Cohesion: 0.09
Nodes (22): resample(), augment(), build_dataset(), confusion(), cross_confusion_rate(), drop_actor_b(), extract_clip(), flush() (+14 more)

### Community 4 - "test_action_pipeline.py"
Cohesion: 0.10
Nodes (24): WeaponDetection, AnomalyType, cap_per_class(), window_index(), calm_pair_sequence(), fall_sequence(), main(), make_person() (+16 more)

### Community 5 - "evaluate.py"
Cohesion: 0.06
Nodes (14): bench_fps(), compare_methods(), confusion(), extract_clip_features(), iter_labelled_clips(), LoadedModel, main(), print_report() (+6 more)

### Community 6 - "people.tsx"
Cohesion: 0.29
Nodes (13): errorMessage(), initials(), LabeledInput(), metaLine(), PeopleScreen(), PersonRow(), PersonSkeleton(), styles (+5 more)

### Community 7 - "[id].tsx"
Cohesion: 0.12
Nodes (32): FaceRow(), Hero(), IconName, Identity(), IncidentDetailScreen(), joinList(), Row(), SectionCard() (+24 more)

### Community 8 - "train_weapon.py"
Cohesion: 0.05
Nodes (28): _cpu_count(), get_profile(), get_yolo_kwargs(), TrainProfile, authenticate(), check_kaggle_json(), download(), _print_token_instructions() (+20 more)

### Community 9 - "dependencies"
Cohesion: 0.07
Nodes (27): dependencies, expo, expo-asset, expo-blur, expo-camera, expo-constants, expo-image-picker, expo-linear-gradient (+19 more)

### Community 10 - "incidents.tsx"
Cohesion: 0.15
Nodes (25): DroneScreen(), dotColorFor(), FilterChip(), FilterKey, FILTERS, IncidentsScreen(), styles, FeedPill() (+17 more)

### Community 11 - "expo"
Cohesion: 0.08
Nodes (24): package, typedRoutes, expo, android, backgroundColor, experiments, ios, name (+16 more)

### Community 12 - "settings.tsx"
Cohesion: 0.16
Nodes (13): CONNECTION_META, HEALTH_LABELS, healthLabel(), HealthRow(), SettingsCard(), SettingsScreen(), styles, useHealth() (+5 more)

### Community 13 - "test_pipeline_shutdown.py"
Cohesion: 0.18
Nodes (9): _now(), main(), make_pipeline(), StubHub, test_join_happens_before_close(), worker(), test_slow_thread_does_not_hang_shutdown(), test_stop_is_idempotent_and_restartable() (+1 more)

### Community 14 - "IncidentStore"
Cohesion: 0.15
Nodes (3): Incident, IncidentStatus, IncidentStore

### Community 15 - "ActionClassifier"
Cohesion: 0.16
Nodes (6): ActionClassifier, label_to_type(), build_model(), build_sequence(), standardize(), test_checkpoint_roundtrip_through_live_wrapper()

### Community 16 - "numpy"
Cohesion: 0.10
Nodes (13): aggressive_pose(), box_iou(), leg_horizontalness(), mid(), proximity(), torso_horizontalness(), analyze(), fall_score() (+5 more)

### Community 17 - "numpy"
Cohesion: 0.21
Nodes (10): _actor_frame(), _body_spread(), fit_standardizer(), frame_feature(), _kp(), _mid(), _min_pair_distance(), _norm_kps() (+2 more)

### Community 18 - "drone.tsx"
Cohesion: 0.19
Nodes (11): HeaderBar(), DPadButton(), enter(), IconName, PatrolRoute, ROUTES, styles, Telemetry (+3 more)

### Community 19 - "package.json"
Cohesion: 0.09
Nodes (21): devDependencies, @types/react, typescript, main, name, private, version, expo-asset (+13 more)

### Community 20 - "get"
Cohesion: 0.15
Nodes (11): Architecture, 1. Vision, Machine Learning & Neural Networks, 2. Backend & Server Infrastructure, 3. Frontend & Mobile Operator App (`app/`), 3. Technology Stack: Every Tech Used & What It's For, attendance(), evidence(), health() (+3 more)

### Community 21 - "app.py"
Cohesion: 0.14
Nodes (5): enroll(), incidents(), people(), set_incident_status(), ws()

### Community 22 - "queries.ts"
Cohesion: 0.17
Nodes (11): api, AttendanceList, AttendanceT, DeleteResult, HealthT, IncidentList, INCIDENTS_LIMIT, IncidentT (+3 more)

### Community 23 - "WeaponDetector"
Cohesion: 0.27
Nodes (3): WeaponDetector, test_weapon_detector_degrades_on_missing_file(), test_weapon_label_matching()

### Community 25 - "config.py"
Cohesion: 0.31
Nodes (4): IngestConfig, ReasonConfig, ServerConfig, StorageConfig

### Community 27 - "attendance.tsx"
Cohesion: 0.23
Nodes (16): AttendanceScreen(), AttendanceSkeletonRow(), fullDateLabel(), Hero(), parseIsoDate(), relativeDayLabel(), shiftDate(), StepperButton() (+8 more)

### Community 28 - "test_fire_detector.py"
Cohesion: 0.12
Nodes (10): FireDetector, main(), test_anti_card_rejection(), test_chromatic_flame_verification(), test_fire_label_matching(), test_fire_precedence_over_violence(), test_graceful_degradation(), test_smoke_verification() (+2 more)

### Community 30 - "action_model.py"
Cohesion: 0.13
Nodes (4): 1. Why violence and fall got confused (and the fix), 1b. The modelling problem, ArcMargin, MultiScaleStem

### Community 31 - "types.ts"
Cohesion: 0.13
Nodes (14): ActionRecord, AnomalyType, BBox, FaceMatch, Health, IncidentStatus, Person, Reasoning (+6 more)

### Community 32 - "HawkEye — Indoor Surveillance & Attendance"
Cohesion: 0.20
Nodes (10): Architecture, Configure, Enroll faces (local identity DB), Evaluate, HawkEye — Indoor Surveillance & Attendance, Project layout, Run (camera + app, together), Setup (+2 more)

### Community 33 - "client.ts"
Cohesion: 0.43
Nodes (5): ApiError, apiUrl(), base(), request(), wsUrl()

### Community 34 - "HawkEye: Complete Architecture & Technical Deep Dive"
Cohesion: 0.20
Nodes (9): 1. Executive Summary: What is HawkEye?, 2. Core Architecture: The Fast/Slow Cascade & Multi-Thread Pipeline, 5. Model Training, Datasets & Benchmarking (`scripts/`, `eval/`), 7. Project Directory Structure Map, 8. Summary of Unique Selling Points (USPs), HawkEye: Complete Architecture & Technical Deep Dive, Multi-Threading Design, The Dual-Mission Model (+1 more)

### Community 35 - "config.py"
Cohesion: 0.06
Nodes (13): Data flow, Design choices, HawkEye — Architecture, Modules, IdentityConfig, FaceMatch, new_id(), Person (+5 more)

### Community 37 - "app/_layout.tsx"
Cohesion: 0.17
Nodes (15): AppShell(), queryClient, RootLayout(), styles, AttendanceRecord, WsEvent, ToastHost(), ToastRow() (+7 more)

### Community 38 - "Training Guide — improving the models"
Cohesion: 0.15
Nodes (13): 0.1 Get the repo, 0.2 Set up the Python environment, 0.2b Run the self-checks first, 0.3 Set up the Kaggle API token (for automatic dataset download), 0.4 Get the datasets, 0.5 Train, then commit + push your CODE changes, 0.6 Training on Windows + NVIDIA CUDA, 0. Set up your environment (+5 more)

### Community 39 - "settings.ts"
Cohesion: 0.31
Nodes (7): DEFAULT_BASE, load(), normalize(), persist(), SettingsState, useSettings, expo-secure-store

### Community 40 - "typing"
Cohesion: 0.12
Nodes (3): is_valid_human_detection(), PoseDetector, resolve_device()

### Community 41 - "HawkEye — Ops App"
Cohesion: 0.22
Nodes (6): A physical phone on the same Wi-Fi, HawkEye — Ops App, Iterating on the UI, One address (the normal way), Running, Screens

### Community 42 - "_index_response"
Cohesion: 0.33
Nodes (3): index(), _index_response(), spa_fallback()

### Community 43 - "runtime.py"
Cohesion: 0.25
Nodes (8): Config, load_config(), main(), get_config(), get_hub(), set_config(), setup_logging(), lifespan()

### Community 44 - "HawkEye App — Context Snapshot"
Cohesion: 0.25
Nodes (7): App architecture, Design decisions (locked, from `/model` plan-mode session), HawkEye App — Context Snapshot, How to run, Key conventions to preserve when extending, Not done yet (future phases, per the plan file), Status: Phase 1 complete

### Community 45 - "scripts"
Cohesion: 0.25
Nodes (8): scripts, android, build:web, ios, lint, start, typecheck, web

### Community 46 - "delete"
Cohesion: 0.33
Nodes (3): clear_incidents(), delete_person(), delete_single_incident()

### Community 47 - "tsconfig.json"
Cohesion: 0.25
Nodes (7): compilerOptions, baseUrl, paths, strict, extends, include, expo/tsconfig.base

### Community 48 - "4. Deep Dive into System Subsystems"
Cohesion: 0.17
Nodes (12): 4.1 Ingest Layer (`hawkeye/ingest/`), 4.2 Stage A: Real-Time Detection & Kinematics (`hawkeye/detect/`), 4.3 Stage B: Vision-Language Model Reasoning (`hawkeye/reason/`), 4.4 Biometric Identity & Passive Attendance (`hawkeye/identity/` & `hawkeye/attendance/`), 4.5 Gated Decision Engine & Dispatch (`hawkeye/decision/`), 4.6 Server, Streaming & WebSocket Protocol (`server/app.py`), 4.7 Frontend Companion Console (`app/`), 4. Deep Dive into System Subsystems (+4 more)

### Community 51 - "2. Get the datasets"
Cohesion: 0.50
Nodes (4): 2. Get the datasets, 2a. Automated (Kaggle) — one command, 2b. Manual (non-Kaggle sources) — optional accuracy boost, Layout & balance

### Community 52 - "(tabs)/_layout.tsx"
Cohesion: 0.67
Nodes (3): TabsLayout(), AnimatedTabBar(), TabItem()

### Community 53 - "HawkEye — Ethics, Privacy & Legal (India)"
Cohesion: 0.40
Nodes (4): HawkEye — Ethics, Privacy & Legal (India), Known limitations to state honestly (thesis), Phase 3 commitments (production), What we do

### Community 54 - "HawkEye — Setup"
Cohesion: 0.40
Nodes (4): Camera options, HawkEye — Setup, macOS (development + training), Reasoner backends

### Community 56 - "metro.config.js"
Cohesion: 0.50
Nodes (3): config, { getDefaultConfig }, expo

### Community 66 - "3. Train the action classifier"
Cohesion: 0.50
Nodes (4): 3. Train the action classifier, Balanced vs. imbalanced — running both off one cache, Keeping the GPU busy, Reading the output (in order of how much it matters)

## Knowledge Gaps
- **13 isolated node(s):** `@expo/metro-runtime`, `expo-asset`, `expo-camera`, `expo-constants`, `expo-linking` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 499 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HawkEye App — Context Snapshot` connect `HawkEye App — Context Snapshot` to `StageAEvent`, `[id].tsx`?**
  _High betweenness centrality (0.392) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `PersonDetection` (e.g. with `aggressive_pose()` and `leg_horizontalness()`) actually correct?**
  _`PersonDetection` has 18 INFERRED edges - model-reasoned connections that need verification._
- **What connects `@expo/metro-runtime`, `expo-asset`, `expo-camera` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `StageAEvent` be split into smaller, more focused modules?**
  _Cohesion score 0.05353535353535353 - nodes in this community are weakly interconnected._
- **Why does `What exists in the backend (unmodified, for reference)` connect `StageAEvent` to `config.py`, `HawkEye App — Context Snapshot`?**
  _High betweenness centrality (0.391) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `StageAEvent` (e.g. with `What exists in the backend (unmodified, for reference)` and `4.2 Stage A: Real-Time Detection & Kinematics (`hawkeye/detect/`)`) actually correct?**
  _`StageAEvent` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Should `react-native` be split into smaller, more focused modules?**
  _Cohesion score 0.10463659147869674 - nodes in this community are weakly interconnected._