# Graph Report - HawkEye-shubh-handoff  (2026-10-15)

## Corpus Check
- 104 files · ~61,210 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 3, .backup 1)

## Summary
- 1117 nodes · 2860 edges · 66 communities (50 shown, 16 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 233 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- pipeline.py
- react-native
- numpy
- train_action.py
- test_action_pipeline.py
- evaluate.py
- typing
- [id].tsx
- settings.tsx
- dependencies
- incidents.tsx
- expo
- people.tsx
- test_pipeline_shutdown.py
- Pipeline
- service.py
- index.ts
- action_classifier.py
- train_weapon.py
- package.json
- EmbeddingStore
- app.py
- setup_datasets.py
- Modules
- 0. Set up your environment
- sys
- pathlib
- attendance.tsx
- VideoSource
- StreamHub
- action_model.py
- AnimatedTabBar.tsx
- HawkEye — Indoor Surveillance & Attendance
- attendance
- HawkEye: Complete Architecture & Technical Deep Dive
- device_profile.py
- .__init__
- app/_layout.tsx
- dispatch.py
- json
- get
- authenticate
- client.ts
- argparse
- HawkEye App — Context Snapshot
- scripts
- settings.ts
- tsconfig.json
- 4. Deep Dive into System Subsystems
- useLiveSocket.ts
- ActionNet
- _index_response
- HawkEye — Architecture
- HawkEye — Ethics, Privacy & Legal (India)
- HawkEye — Setup
- enroll
- metro.config.js
- 3. Train the action classifier
- devDependencies
- train_all.sh
- delete_person

## God Nodes (most connected - your core abstractions)
1. `PersonDetection` - 30 edges
2. `StageAEvent` - 28 edges
3. `react-native` - 27 edges
4. `Pipeline` - 27 edges
5. `react` - 26 edges
6. `frame_feature()` - 26 edges
7. `Reasoning` - 26 edges
8. `Palette` - 25 edges
9. `AnomalyType` - 24 edges
10. `StageA` - 22 edges

## Surprising Connections (you probably didn't know these)
- `6. Where each piece lives (quick map)` --references--> `ActionNet`  [INFERRED]
  docs/training.md → hawkeye/detect/action_model.py
- `Solving the Violence vs. Fall Confusion` --references--> `ActionNet`  [INFERRED]
  HAWKEYE_DEEP_DIVE.md → hawkeye/detect/action_model.py
- `5. Train the weapon detector (knife / gun / etc.)` --references--> `convert_voc_to_yolo()`  [INFERRED]
  docs/training.md → scripts/train_weapon.py
- `Modules` --references--> `AttendanceStore`  [INFERRED]
  docs/architecture.md → hawkeye/attendance/store.py
- `Modules` --references--> `DecisionEngine`  [INFERRED]
  docs/architecture.md → hawkeye/decision/engine.py

## Import Cycles
- 3-file cycle: `hawkeye/detect/__init__.py -> hawkeye/detect/action_classifier.py -> hawkeye/detect/skeleton_features.py -> hawkeye/detect/__init__.py`

## Communities (66 total, 16 thin omitted)

### Community 0 - "pipeline.py"
Cohesion: 0.06
Nodes (39): What exists in the backend (unmodified, for reference), Config, DecisionConfig, DetectConfig, IngestConfig, ReasonConfig, ServerConfig, StorageConfig (+31 more)

### Community 1 - "react-native"
Cohesion: 0.08
Nodes (41): CONNECTION_META, Props, styles, Props, styles, Variant, VARIANTS, Props (+33 more)

### Community 2 - "numpy"
Cohesion: 0.10
Nodes (18): aggressive_pose(), box_iou(), mid(), proximity(), torso_horizontalness(), analyze(), fall_score(), PersonDetection (+10 more)

### Community 3 - "train_action.py"
Cohesion: 0.08
Nodes (27): 1. Why violence and fall got confused (and the fix), 1a. The two bugs that made a "0.95 F1" model behave randomly, 1b. The modelling problem, resample(), standardize(), augment(), build_dataset(), cap_per_class() (+19 more)

### Community 4 - "test_action_pipeline.py"
Cohesion: 0.09
Nodes (22): build_model(), build_sequence(), fit_standardizer(), window_index(), calm_pair_sequence(), fall_sequence(), main(), make_person() (+14 more)

### Community 5 - "evaluate.py"
Cohesion: 0.10
Nodes (12): bench_fps(), compare_methods(), confusion(), extract_clip_features(), iter_labelled_clips(), LoadedModel, main(), print_report() (+4 more)

### Community 6 - "typing"
Cohesion: 0.10
Nodes (5): MotionAnomaly, StageA, WeaponDetection, WeaponDetector, BBox

### Community 7 - "[id].tsx"
Cohesion: 0.15
Nodes (27): FaceRow(), HeaderBar(), Hero(), IconName, Identity(), IncidentDetailScreen(), joinList(), Row() (+19 more)

### Community 8 - "settings.tsx"
Cohesion: 0.15
Nodes (22): CONNECTION_META, HEALTH_LABELS, healthLabel(), HealthRow(), SettingsCard(), SettingsScreen(), styles, DPadButton() (+14 more)

### Community 9 - "dependencies"
Cohesion: 0.07
Nodes (27): dependencies, expo, expo-asset, expo-blur, expo-camera, expo-constants, expo-image-picker, expo-linear-gradient (+19 more)

### Community 10 - "incidents.tsx"
Cohesion: 0.18
Nodes (23): dotColorFor(), FilterChip(), FilterKey, FILTERS, IncidentsScreen(), styles, FeedPill(), fmtFps() (+15 more)

### Community 11 - "expo"
Cohesion: 0.08
Nodes (24): package, typedRoutes, expo, android, backgroundColor, experiments, ios, name (+16 more)

### Community 12 - "people.tsx"
Cohesion: 0.12
Nodes (23): errorMessage(), initials(), LabeledInput(), metaLine(), PeopleScreen(), PersonRow(), PersonSkeleton(), styles (+15 more)

### Community 13 - "test_pipeline_shutdown.py"
Cohesion: 0.14
Nodes (9): main(), make_pipeline(), StubHub, StubStore, test_join_happens_before_close(), worker(), test_slow_thread_does_not_hang_shutdown(), test_stop_is_idempotent_and_restartable() (+1 more)

### Community 14 - "Pipeline"
Cohesion: 0.13
Nodes (3): Incident, IncidentStore, Pipeline

### Community 15 - "service.py"
Cohesion: 0.13
Nodes (3): IdentityConfig, Face, IdentityRecognizer

### Community 16 - "index.ts"
Cohesion: 0.14
Nodes (17): ActionRecord, ActionType, AnomalyType, BBox, FaceMatch, IncidentStatus, Reasoning, Severity (+9 more)

### Community 18 - "train_weapon.py"
Cohesion: 0.16
Nodes (9): _now(), auto_download_weapons(), _build_yolo_from_voc(), convert_voc_to_yolo(), create_weapons_yaml(), main(), train(), verify_dataset() (+1 more)

### Community 19 - "package.json"
Cohesion: 0.11
Nodes (18): main, name, private, version, expo-asset, expo-blur, expo-camera, expo-constants (+10 more)

### Community 21 - "app.py"
Cohesion: 0.16
Nodes (5): get_config(), get_hub(), setup_logging(), lifespan(), ws()

### Community 22 - "setup_datasets.py"
Cohesion: 0.22
Nodes (8): check_disk_space(), download_from_kaggle(), ensure_dirs(), main(), organize_from_local(), organize_ucf101_normals(), print_download_instructions(), verify_dataset()

### Community 24 - "0. Set up your environment"
Cohesion: 0.12
Nodes (17): 0.1 Get the repo, 0.2 Set up the Python environment, 0.2b Run the self-checks first, 0.3 Set up the Kaggle API token (for automatic dataset download), 0.4 Get the datasets, 0.5 Train, then commit + push your CODE changes, 0.6 Training on Windows + NVIDIA CUDA, 0. Set up your environment (+9 more)

### Community 25 - "sys"
Cohesion: 0.19
Nodes (4): extract_middle_frame(), main(), main(), validate_clip()

### Community 26 - "pathlib"
Cohesion: 0.17
Nodes (3): AttendanceStore, date_str(), AttendanceRecord

### Community 27 - "attendance.tsx"
Cohesion: 0.29
Nodes (14): AttendanceRow(), AttendanceScreen(), AttendanceSkeletonRow(), fullDateLabel(), Hero(), parseIsoDate(), relativeDayLabel(), shiftDate() (+6 more)

### Community 28 - "VideoSource"
Cohesion: 0.16
Nodes (4): 4.1 Ingest Layer (`hawkeye/ingest/`), DroneSource, VideoSource, Roadmap

### Community 31 - "AnimatedTabBar.tsx"
Cohesion: 0.19
Nodes (8): TabsLayout(), AnimatedTabBar(), ICONS, styles, TabItem(), motion, expo-router, react-native-safe-area-context

### Community 32 - "HawkEye — Indoor Surveillance & Attendance"
Cohesion: 0.15
Nodes (10): Architecture, Configure, Enroll faces (local identity DB), Evaluate, HawkEye — Indoor Surveillance & Attendance, Project layout, Run (camera + app, together), Setup (+2 more)

### Community 33 - "attendance"
Cohesion: 0.17
Nodes (13): A physical phone on the same Wi-Fi, Architecture, HawkEye — Ops App, Iterating on the UI, One address (the normal way), Running, Screens, 2. Backend & Server Infrastructure (+5 more)

### Community 34 - "HawkEye: Complete Architecture & Technical Deep Dive"
Cohesion: 0.15
Nodes (12): 1. Executive Summary: What is HawkEye?, 1. Vision, Machine Learning & Neural Networks, 2. Core Architecture: The Fast/Slow Cascade & Multi-Thread Pipeline, 3. Frontend & Mobile Operator App (`app/`), 3. Technology Stack: Every Tech Used & What It's For, 5. Model Training, Datasets & Benchmarking (`scripts/`, `eval/`), 7. Project Directory Structure Map, 8. Summary of Unique Selling Points (USPs) (+4 more)

### Community 35 - "device_profile.py"
Cohesion: 0.26
Nodes (4): _cpu_count(), get_profile(), get_yolo_kwargs(), TrainProfile

### Community 37 - "app/_layout.tsx"
Cohesion: 0.24
Nodes (10): AppShell(), queryClient, RootLayout(), styles, ToastHost(), ToastRow(), useToast, expo-status-bar (+2 more)

### Community 38 - "dispatch.py"
Cohesion: 0.27
Nodes (4): build_dispatchers(), Dispatcher, LogDispatcher, WebhookDispatcher

### Community 40 - "get"
Cohesion: 0.20
Nodes (6): evidence(), health(), incidents(), incidents_csv(), people(), video_feed()

### Community 41 - "authenticate"
Cohesion: 0.29
Nodes (4): authenticate(), check_kaggle_json(), download(), _print_token_instructions()

### Community 42 - "client.ts"
Cohesion: 0.33
Nodes (6): api, ApiError, apiUrl(), base(), request(), zod

### Community 43 - "argparse"
Cohesion: 0.31
Nodes (3): load_config(), main(), set_config()

### Community 44 - "HawkEye App — Context Snapshot"
Cohesion: 0.25
Nodes (7): App architecture, Design decisions (locked, from `/model` plan-mode session), HawkEye App — Context Snapshot, How to run, Key conventions to preserve when extending, Not done yet (future phases, per the plan file), Status: Phase 1 complete

### Community 45 - "scripts"
Cohesion: 0.25
Nodes (8): scripts, android, build:web, ios, lint, start, typecheck, web

### Community 46 - "settings.ts"
Cohesion: 0.32
Nodes (6): DEFAULT_BASE, load(), persist(), SettingsState, useSettings, expo-secure-store

### Community 47 - "tsconfig.json"
Cohesion: 0.25
Nodes (7): compilerOptions, baseUrl, paths, strict, extends, include, expo/tsconfig.base

### Community 48 - "4. Deep Dive into System Subsystems"
Cohesion: 0.25
Nodes (8): 4.2 Stage A: Real-Time Detection & Kinematics (`hawkeye/detect/`), 4.3 Stage B: Vision-Language Model Reasoning (`hawkeye/reason/`), 4.5 Gated Decision Engine & Dispatch (`hawkeye/decision/`), 4.6 Server, Streaming & WebSocket Protocol (`server/app.py`), 4.7 Frontend Companion Console (`app/`), 4. Deep Dive into System Subsystems, Solving the Violence vs. Fall Confusion, The 188-Dimensional Skeleton Feature Representation (`skeleton_features.py`)

### Community 49 - "useLiveSocket.ts"
Cohesion: 0.38
Nodes (6): wsUrl(), AttendanceRecord, WsEvent, arrayBufferToBase64(), useLiveSocket(), @tanstack/react-query

### Community 51 - "_index_response"
Cohesion: 0.33
Nodes (3): index(), _index_response(), spa_fallback()

### Community 52 - "HawkEye — Architecture"
Cohesion: 0.40
Nodes (4): Data flow, Design choices, Extension points, HawkEye — Architecture

### Community 53 - "HawkEye — Ethics, Privacy & Legal (India)"
Cohesion: 0.40
Nodes (4): HawkEye — Ethics, Privacy & Legal (India), Known limitations to state honestly (thesis), Phase 3 commitments (production), What we do

### Community 54 - "HawkEye — Setup"
Cohesion: 0.40
Nodes (4): Camera options, HawkEye — Setup, macOS (development + training), Reasoner backends

### Community 56 - "metro.config.js"
Cohesion: 0.50
Nodes (3): config, { getDefaultConfig }, expo

### Community 57 - "3. Train the action classifier"
Cohesion: 0.50
Nodes (4): 3. Train the action classifier, Balanced vs. imbalanced — running both off one cache, Keeping the GPU busy, Reading the output (in order of how much it matters)

### Community 58 - "devDependencies"
Cohesion: 0.67
Nodes (3): devDependencies, @types/react, typescript

## Knowledge Gaps
- **13 isolated node(s):** `@expo/metro-runtime`, `expo-asset`, `expo-camera`, `expo-constants`, `expo-linking` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 468 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HawkEye App — Context Snapshot` connect `HawkEye App — Context Snapshot` to `pipeline.py`, `attendance.tsx`?**
  _High betweenness centrality (0.392) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `PersonDetection` (e.g. with `aggressive_pose()` and `proximity()`) actually correct?**
  _`PersonDetection` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `@expo/metro-runtime`, `expo-asset`, `expo-camera` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `pipeline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.058435255290234904 - nodes in this community are weakly interconnected._
- **Why does `What exists in the backend (unmodified, for reference)` connect `pipeline.py` to `HawkEye App — Context Snapshot`?**
  _High betweenness centrality (0.391) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `StageAEvent` (e.g. with `What exists in the backend (unmodified, for reference)` and `4.2 Stage A: Real-Time Detection & Kinematics (`hawkeye/detect/`)`) actually correct?**
  _`StageAEvent` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Should `react-native` be split into smaller, more focused modules?**
  _Cohesion score 0.08196721311475409 - nodes in this community are weakly interconnected._