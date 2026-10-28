# Graph Report - HawkEye-shubh-handoff  (2026-10-28)

## Corpus Check
- 105 files · ~72,681 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: .pt 4, (none) 3)

## Summary
- 1153 nodes · 2977 edges · 56 communities (46 shown, 10 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 255 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f384b786`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- StageAEvent
- react-native
- StageA
- test_action_pipeline.py
- test_action_pipeline.py
- evaluate.py
- MotionAnomaly
- [id].tsx
- setup_datasets.py
- dependencies
- incidents.tsx
- expo
- settings.tsx
- test_pipeline_shutdown.py
- pipeline.py
- ActionClassifier
- numpy
- action_classifier.py
- train_weapon.py
- package.json
- EmbeddingStore
- app.py
- index.ts
- sys
- device_profile.py
- authenticate
- attendance/store.py
- attendance.tsx
- WeaponDetector
- StreamHub
- action_model.py
- (tabs)/_layout.tsx
- HawkEye — Indoor Surveillance & Attendance
- HawkEye: Complete Architecture & Technical Deep Dive
- Modules
- Pipeline
- react
- Training Guide — improving the models
- HawkEye — Ops App
- config.py
- HawkEye App — Context Snapshot
- scripts
- tsconfig.json
- 4. Deep Dive into System Subsystems
- ActionNet
- HawkEye — Ethics, Privacy & Legal (India)
- HawkEye — Setup
- metro.config.js
- train_all.sh
- 3. Train the action classifier
- devDependencies

## God Nodes (most connected - your core abstractions)
1. `PersonDetection` - 35 edges
2. `StageAEvent` - 31 edges
3. `AnomalyType` - 28 edges
4. `react-native` - 27 edges
5. `Pipeline` - 27 edges
6. `react` - 26 edges
7. `frame_feature()` - 26 edges
8. `StageA` - 26 edges
9. `Reasoning` - 26 edges
10. `Palette` - 25 edges

## Surprising Connections (you probably didn't know these)
- `6. Where each piece lives (quick map)` --references--> `ActionNet`  [INFERRED]
  docs/training.md → hawkeye/detect/action_model.py
- `Solving the Violence vs. Fall Confusion` --references--> `ActionNet`  [INFERRED]
  HAWKEYE_DEEP_DIVE.md → hawkeye/detect/action_model.py
- `Continuous Passive Attendance (`hawkeye/attendance/`)` --references--> `attendance()`  [INFERRED]
  HAWKEYE_DEEP_DIVE.md → server/app.py
- `Modules` --references--> `AttendanceStore`  [INFERRED]
  docs/architecture.md → hawkeye/attendance/store.py
- `Modules` --references--> `DecisionEngine`  [INFERRED]
  docs/architecture.md → hawkeye/decision/engine.py

## Import Cycles
- 3-file cycle: `hawkeye/detect/__init__.py -> hawkeye/detect/action_classifier.py -> hawkeye/detect/skeleton_features.py -> hawkeye/detect/__init__.py`
- 3-file cycle: `hawkeye/detect/__init__.py -> hawkeye/detect/action_classifier.py -> hawkeye/detect/heuristics.py -> hawkeye/detect/__init__.py`

## Communities (56 total, 10 thin omitted)

### Community 0 - "StageAEvent"
Cohesion: 0.05
Nodes (36): What exists in the backend (unmodified, for reference), DecisionConfig, build_dispatchers(), Dispatcher, LogDispatcher, WebhookDispatcher, DecisionEngine, ActionRecord (+28 more)

### Community 1 - "react-native"
Cohesion: 0.08
Nodes (42): enter(), IconName, PatrolRoute, ROUTES, styles, Telemetry, ICONS, styles (+34 more)

### Community 2 - "StageA"
Cohesion: 0.22
Nodes (3): box_iou(), StageA, BBox

### Community 3 - "test_action_pipeline.py"
Cohesion: 0.07
Nodes (28): 1. Why violence and fall got confused (and the fix), 1a. The two bugs that made a "0.95 F1" model behave randomly, 1b. The modelling problem, fit_standardizer(), resample(), standardize(), augment(), build_dataset() (+20 more)

### Community 4 - "test_action_pipeline.py"
Cohesion: 0.11
Nodes (23): DetectConfig, build_sequence(), WeaponDetection, calm_pair_sequence(), fall_sequence(), main(), make_person(), test_calm_pair_is_not_violent_looking() (+15 more)

### Community 5 - "evaluate.py"
Cohesion: 0.07
Nodes (14): bench_fps(), compare_methods(), confusion(), extract_clip_features(), iter_labelled_clips(), LoadedModel, main(), print_report() (+6 more)

### Community 6 - "MotionAnomaly"
Cohesion: 0.13
Nodes (7): Data flow, Design choices, Extension points, HawkEye — Architecture, MotionAnomaly, extract_clip(), flush()

### Community 7 - "[id].tsx"
Cohesion: 0.15
Nodes (27): FaceRow(), HeaderBar(), Hero(), IconName, Identity(), IncidentDetailScreen(), joinList(), Row() (+19 more)

### Community 8 - "setup_datasets.py"
Cohesion: 0.22
Nodes (8): check_disk_space(), download_from_kaggle(), ensure_dirs(), main(), organize_from_local(), organize_ucf101_normals(), print_download_instructions(), verify_dataset()

### Community 9 - "dependencies"
Cohesion: 0.07
Nodes (27): dependencies, expo, expo-asset, expo-blur, expo-camera, expo-constants, expo-image-picker, expo-linear-gradient (+19 more)

### Community 10 - "incidents.tsx"
Cohesion: 0.14
Nodes (33): DroneScreen(), dotColorFor(), FilterChip(), FilterKey, FILTERS, IncidentsScreen(), styles, FeedPill() (+25 more)

### Community 11 - "expo"
Cohesion: 0.08
Nodes (24): package, typedRoutes, expo, android, backgroundColor, experiments, ios, name (+16 more)

### Community 12 - "settings.tsx"
Cohesion: 0.07
Nodes (37): CONNECTION_META, HEALTH_LABELS, healthLabel(), HealthRow(), SettingsCard(), SettingsScreen(), styles, AttendanceList (+29 more)

### Community 13 - "test_pipeline_shutdown.py"
Cohesion: 0.14
Nodes (9): main(), make_pipeline(), StubHub, StubStore, test_join_happens_before_close(), worker(), test_slow_thread_does_not_hang_shutdown(), test_stop_is_idempotent_and_restartable() (+1 more)

### Community 14 - "pipeline.py"
Cohesion: 0.11
Nodes (3): Incident, IncidentStatus, IncidentStore

### Community 15 - "ActionClassifier"
Cohesion: 0.22
Nodes (3): ActionClassifier, build_model(), test_checkpoint_roundtrip_through_live_wrapper()

### Community 16 - "numpy"
Cohesion: 0.10
Nodes (21): aggressive_pose(), leg_horizontalness(), mid(), proximity(), torso_horizontalness(), analyze(), fall_score(), is_human() (+13 more)

### Community 17 - "action_classifier.py"
Cohesion: 0.12
Nodes (4): label_to_type(), is_valid_human_detection(), PoseDetector, resolve_device()

### Community 18 - "train_weapon.py"
Cohesion: 0.15
Nodes (10): 5. Train the weapon detector (knife / gun / etc.), _now(), auto_download_weapons(), _build_yolo_from_voc(), convert_voc_to_yolo(), create_weapons_yaml(), main(), train() (+2 more)

### Community 19 - "package.json"
Cohesion: 0.10
Nodes (20): main, name, private, version, expo-asset, expo-blur, expo-camera, expo-constants (+12 more)

### Community 20 - "EmbeddingStore"
Cohesion: 0.15
Nodes (3): new_id(), Person, EmbeddingStore

### Community 21 - "app.py"
Cohesion: 0.05
Nodes (25): Architecture, 1. Vision, Machine Learning & Neural Networks, 2. Backend & Server Infrastructure, 3. Frontend & Mobile Operator App (`app/`), 3. Technology Stack: Every Tech Used & What It's For, get_config(), get_hub(), attendance() (+17 more)

### Community 22 - "index.ts"
Cohesion: 0.19
Nodes (13): ActionRecord, ActionType, AnomalyType, BBox, FaceMatch, IncidentStatus, Reasoning, Severity (+5 more)

### Community 23 - "sys"
Cohesion: 0.20
Nodes (4): extract_middle_frame(), main(), main(), validate_clip()

### Community 24 - "device_profile.py"
Cohesion: 0.26
Nodes (4): _cpu_count(), get_profile(), get_yolo_kwargs(), TrainProfile

### Community 25 - "authenticate"
Cohesion: 0.29
Nodes (4): authenticate(), check_kaggle_json(), download(), _print_token_instructions()

### Community 26 - "attendance/store.py"
Cohesion: 0.15
Nodes (3): AttendanceStore, date_str(), AttendanceRecord

### Community 27 - "attendance.tsx"
Cohesion: 0.17
Nodes (20): AttendanceRow(), AttendanceScreen(), AttendanceSkeletonRow(), fullDateLabel(), Hero(), parseIsoDate(), relativeDayLabel(), shiftDate() (+12 more)

### Community 31 - "(tabs)/_layout.tsx"
Cohesion: 0.33
Nodes (4): TabsLayout(), AnimatedTabBar(), TabItem(), expo-router

### Community 32 - "HawkEye — Indoor Surveillance & Attendance"
Cohesion: 0.20
Nodes (10): Architecture, Configure, Enroll faces (local identity DB), Evaluate, HawkEye — Indoor Surveillance & Attendance, Project layout, Run (camera + app, together), Setup (+2 more)

### Community 34 - "HawkEye: Complete Architecture & Technical Deep Dive"
Cohesion: 0.20
Nodes (9): 1. Executive Summary: What is HawkEye?, 2. Core Architecture: The Fast/Slow Cascade & Multi-Thread Pipeline, 5. Model Training, Datasets & Benchmarking (`scripts/`, `eval/`), 7. Project Directory Structure Map, 8. Summary of Unique Selling Points (USPs), HawkEye: Complete Architecture & Technical Deep Dive, Multi-Threading Design, The Dual-Mission Model (+1 more)

### Community 35 - "Modules"
Cohesion: 0.11
Nodes (4): Modules, Face, IdentityRecognizer, IdentityService

### Community 36 - "Pipeline"
Cohesion: 0.14
Nodes (3): 6. Ethics, Privacy & Legal Framework (DPDP Act, 2023), EvidenceManager, Pipeline

### Community 37 - "react"
Cohesion: 0.11
Nodes (25): AppShell(), queryClient, RootLayout(), styles, api, ApiError, apiUrl(), base() (+17 more)

### Community 38 - "Training Guide — improving the models"
Cohesion: 0.12
Nodes (16): 0.1 Get the repo, 0.2 Set up the Python environment, 0.2b Run the self-checks first, 0.3 Set up the Kaggle API token (for automatic dataset download), 0.4 Get the datasets, 0.5 Train, then commit + push your CODE changes, 0.6 Training on Windows + NVIDIA CUDA, 0. Set up your environment (+8 more)

### Community 41 - "HawkEye — Ops App"
Cohesion: 0.22
Nodes (6): A physical phone on the same Wi-Fi, HawkEye — Ops App, Iterating on the UI, One address (the normal way), Running, Screens

### Community 43 - "config.py"
Cohesion: 0.13
Nodes (11): Config, IdentityConfig, IngestConfig, load_config(), ReasonConfig, ServerConfig, StorageConfig, main() (+3 more)

### Community 44 - "HawkEye App — Context Snapshot"
Cohesion: 0.25
Nodes (7): App architecture, Design decisions (locked, from `/model` plan-mode session), HawkEye App — Context Snapshot, How to run, Key conventions to preserve when extending, Not done yet (future phases, per the plan file), Status: Phase 1 complete

### Community 45 - "scripts"
Cohesion: 0.25
Nodes (8): scripts, android, build:web, ios, lint, start, typecheck, web

### Community 47 - "tsconfig.json"
Cohesion: 0.25
Nodes (7): compilerOptions, baseUrl, paths, strict, extends, include, expo/tsconfig.base

### Community 48 - "4. Deep Dive into System Subsystems"
Cohesion: 0.17
Nodes (12): 4.1 Ingest Layer (`hawkeye/ingest/`), 4.2 Stage A: Real-Time Detection & Kinematics (`hawkeye/detect/`), 4.3 Stage B: Vision-Language Model Reasoning (`hawkeye/reason/`), 4.4 Biometric Identity & Passive Attendance (`hawkeye/identity/` & `hawkeye/attendance/`), 4.5 Gated Decision Engine & Dispatch (`hawkeye/decision/`), 4.6 Server, Streaming & WebSocket Protocol (`server/app.py`), 4.7 Frontend Companion Console (`app/`), 4. Deep Dive into System Subsystems (+4 more)

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

### Community 67 - "devDependencies"
Cohesion: 0.67
Nodes (3): devDependencies, @types/react, typescript

## Knowledge Gaps
- **13 isolated node(s):** `@expo/metro-runtime`, `expo-asset`, `expo-camera`, `expo-constants`, `expo-linking` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 483 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HawkEye App — Context Snapshot` connect `HawkEye App — Context Snapshot` to `StageAEvent`, `attendance.tsx`?**
  _High betweenness centrality (0.396) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `PersonDetection` (e.g. with `aggressive_pose()` and `leg_horizontalness()`) actually correct?**
  _`PersonDetection` has 18 INFERRED edges - model-reasoned connections that need verification._
- **What connects `@expo/metro-runtime`, `expo-asset`, `expo-camera` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `StageAEvent` be split into smaller, more focused modules?**
  _Cohesion score 0.05483405483405483 - nodes in this community are weakly interconnected._
- **Why does `What exists in the backend (unmodified, for reference)` connect `StageAEvent` to `HawkEye App — Context Snapshot`?**
  _High betweenness centrality (0.395) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `StageAEvent` (e.g. with `What exists in the backend (unmodified, for reference)` and `4.2 Stage A: Real-Time Detection & Kinematics (`hawkeye/detect/`)`) actually correct?**
  _`StageAEvent` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Should `react-native` be split into smaller, more focused modules?**
  _Cohesion score 0.08249603384452671 - nodes in this community are weakly interconnected._