/**
 * Camera & Video Source — turns your phone into an intelligent surveillance eye
 * or switches between real-time video sources (phone camera, screen capture,
 * webcam, or IP camera RTSP streams).
 */
import React, { useEffect, useRef, useState } from 'react';
import {
  ScrollView,
  Text,
  View,
  StyleSheet,
  TextInput,
  Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown } from 'react-native-reanimated';

import { palette, spacing, typography, radii, motion } from '@/theme';
import { Screen } from '@/components/Screen';
import { AppHeader } from '@/components/AppHeader';
import { ConnectionBanner } from '@/components/ConnectionBanner';
import { Card } from '@/components/Card';
import { Badge } from '@/components/Badge';
import { Button } from '@/components/Button';
import { SectionHeader } from '@/components/SectionHeader';
import { StatusDot } from '@/components/StatusDot';
import { PressableScale } from '@/components/PressableScale';
import { toast } from '@/stores/toast';
import { useSettings } from '@/stores/settings';
import { useCameraStream } from '@/stores/cameraStream';
import { useCameraStatus, useSetCameraSource } from '@/api/queries';

const TAB_BAR_SPACE = 108;

function enter(index: number) {
  return FadeInDown.delay(Math.min(index, 8) * motion.stagger)
    .springify()
    .damping(18);
}

export default function CameraScreen() {
  const router = useRouter();
  const serverBase = useSettings((s) => s.serverBase);
  const { data: camStatus } = useCameraStatus();
  const setSource = useSetCameraSource();

  const {
    isCameraActive,
    isBroadcasting,
    facingMode,
    fps,
    framesSent,
    error: streamError,
    startCamera,
    stopCamera,
    toggleFacing,
    startBroadcasting,
    stopBroadcasting,
    attachVideoElement,
  } = useCameraStream();

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [customUrl, setCustomUrl] = useState('');
  const [switching, setSwitching] = useState(false);

  // Mount video element ref to store
  useEffect(() => {
    if (Platform.OS === 'web' && videoRef.current) {
      attachVideoElement(videoRef.current);
    }
  }, [attachVideoElement]);

  // Auto-start camera preview on mount if web
  useEffect(() => {
    if (Platform.OS === 'web' && !isCameraActive) {
      startCamera();
    }
  }, []);

  const handleToggleBroadcast = () => {
    if (isBroadcasting) {
      stopBroadcasting();
      toast.show('Broadcasting stopped', 'info', 'radio');
    } else {
      startBroadcasting(serverBase);
      toast.show('Broadcasting phone camera to HawkEye', 'success', 'radio');
    }
  };

  const handleSwitchSource = async (src: string) => {
    setSwitching(true);
    try {
      await setSource.mutateAsync(src);
      toast.show(`Switched source to ${src}`, 'success', 'checkmark-circle');
    } catch {
      toast.show('Failed to switch video source', 'alert', 'warning');
    } finally {
      setSwitching(false);
    }
  };

  const handleConnectCustom = async () => {
    const url = customUrl.trim();
    if (!url) {
      toast.show('Enter a valid URL or RTSP stream', 'info', 'alert-circle');
      return;
    }
    await handleSwitchSource(url);
    setCustomUrl('');
  };

  const currentSource = camStatus?.current_source ?? '0';
  const isPhoneActive = currentSource === 'phone' || isBroadcasting;
  const connectUrl = camStatus?.connect_url || serverBase;

  const copyConnectUrl = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(connectUrl);
      toast.show('Link copied to clipboard', 'info', 'copy');
    } else {
      toast.show(connectUrl, 'info', 'link');
    }
  };

  return (
    <Screen>
      <AppHeader
        title="Camera"
        subtitle="Source & Phone Streaming"
        right={
          <PressableScale onPress={() => router.push('/settings')} style={styles.gear}>
            <Ionicons name="settings-outline" size={18} color={palette.textSecondary} />
          </PressableScale>
        }
      />
      <ConnectionBanner />

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.content}>
        {/* --- Viewfinder Card --- */}
        <Animated.View entering={enter(0)}>
          <Card elevated style={styles.viewfinderCard}>
            <View style={styles.viewfinderContainer}>
              {Platform.OS === 'web' ? (
                <video
                  ref={(el) => {
                    videoRef.current = el;
                    attachVideoElement(el);
                  }}
                  autoPlay
                  playsInline
                  muted
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    backgroundColor: '#121214',
                  }}
                />
              ) : (
                <View style={styles.fallbackBox}>
                  <Ionicons name="videocam-outline" size={48} color={palette.textTertiary} />
                  <Text style={styles.fallbackText}>Camera available in web view</Text>
                </View>
              )}

              {/* Viewfinder Overlays & Crosshairs */}
              <View style={styles.hudOverlay} pointerEvents="box-none">
                <View style={styles.hudHeader}>
                  <View style={styles.hudBadgeGroup}>
                    <Badge
                      label={isBroadcasting ? 'BROADCASTING' : isCameraActive ? 'VIEWFINDER' : 'STANDBY'}
                      fg={isBroadcasting ? palette.green : isCameraActive ? palette.cyan : palette.amber}
                      bg={isBroadcasting ? palette.greenDim : isCameraActive ? palette.cyanDim : palette.amberDim}
                    />
                    <Badge
                      label={facingMode === 'environment' ? 'Back Lens' : 'Front Lens'}
                      fg={palette.textSecondary}
                      bg={palette.surfaceElevated}
                    />
                  </View>
                  {isBroadcasting && (
                    <View style={styles.liveIndicator}>
                      <StatusDot color={palette.green} size={8} glow />
                      <Text style={styles.liveFpsText}>{fps > 0 ? `${fps} FPS` : 'Streaming'}</Text>
                    </View>
                  )}
                </View>

                {/* Tactical Framing Corners */}
                <View style={[styles.corner, styles.cornerTL]} />
                <View style={[styles.corner, styles.cornerTR]} />
                <View style={[styles.corner, styles.cornerBL]} />
                <View style={[styles.corner, styles.cornerBR]} />

                {/* Viewfinder Bottom Controls */}
                <View style={styles.hudFooter}>
                  <PressableScale
                    onPress={toggleFacing}
                    style={styles.hudBtn}
                    accessibilityLabel="Flip Camera"
                  >
                    <Ionicons name="camera-reverse" size={18} color={palette.text} />
                    <Text style={styles.hudBtnText}>Flip</Text>
                  </PressableScale>

                  {isCameraActive ? (
                    <PressableScale onPress={stopCamera} style={styles.hudBtn}>
                      <Ionicons name="power" size={18} color={palette.red} />
                      <Text style={[styles.hudBtnText, { color: palette.red }]}>Pause</Text>
                    </PressableScale>
                  ) : (
                    <PressableScale onPress={() => startCamera()} style={styles.hudBtn}>
                      <Ionicons name="play" size={18} color={palette.green} />
                      <Text style={[styles.hudBtnText, { color: palette.green }]}>Start</Text>
                    </PressableScale>
                  )}
                </View>
              </View>
            </View>

            {streamError && (
              <View style={styles.errorNotice}>
                <Ionicons name="alert-circle" size={16} color={palette.red} />
                <Text style={styles.errorNoticeText}>{streamError}</Text>
              </View>
            )}

            {/* Broadcast Action Row */}
            <View style={styles.broadcastActionWrap}>
              <Button
                variant={isBroadcasting ? 'danger' : 'primary'}
                label={isBroadcasting ? 'Stop Broadcasting' : 'Broadcast to HawkEye'}
                icon={isBroadcasting ? 'stop-circle' : 'radio'}
                onPress={handleToggleBroadcast}
                fullWidth
              />
              <Text style={styles.broadcastSubtext}>
                {isBroadcasting
                  ? 'Your phone is currently serving as the live camera feed for HawkEye.'
                  : 'Transmit this camera feed over WebSocket into the AI surveillance pipeline.'}
              </Text>
            </View>
          </Card>
        </Animated.View>

        {/* --- Telemetry Strip --- */}
        {isBroadcasting && (
          <Animated.View entering={enter(1)}>
            <View style={styles.statRow}>
              <View style={styles.statTile}>
                <Text style={styles.statLabel}>STREAM FPS</Text>
                <Text style={[styles.statValue, { color: palette.green }]}>
                  {fps.toFixed(1)}
                </Text>
              </View>
              <View style={styles.statTile}>
                <Text style={styles.statLabel}>FRAMES PUSHED</Text>
                <Text style={styles.statValue}>{framesSent}</Text>
              </View>
              <View style={styles.statTile}>
                <Text style={styles.statLabel}>PIPELINE</Text>
                <Text style={[styles.statValue, { color: palette.accent }]}>Active</Text>
              </View>
            </View>
          </Animated.View>
        )}

        {/* --- Pipeline Source Switcher --- */}
        <Animated.View entering={enter(2)}>
          <Card padded style={styles.sourceCard}>
            <SectionHeader
              title="Pipeline Video Source"
              right={
                <Badge
                  label={`Current: ${currentSource}`}
                  fg={isPhoneActive ? palette.green : palette.accent}
                  bg={isPhoneActive ? palette.greenDim : palette.accentDim}
                />
              }
            />
            <Text style={styles.sectionCaption}>
              Select which video stream HawkEye's detection and attendance pipeline analyzes.
            </Text>

            <View style={styles.sourceBtnGrid}>
              <PressableScale
                onPress={() => handleSwitchSource('phone')}
                style={[
                  styles.sourceOption,
                  currentSource === 'phone' && styles.sourceOptionActive,
                ]}
              >
                <Ionicons
                  name="phone-portrait-outline"
                  size={22}
                  color={currentSource === 'phone' ? palette.accent : palette.textSecondary}
                />
                <Text
                  style={[
                    styles.sourceOptionTitle,
                    currentSource === 'phone' && styles.sourceOptionTitleActive,
                  ]}
                >
                  Phone Camera
                </Text>
                <Text style={styles.sourceOptionDesc}>Stream via browser</Text>
              </PressableScale>

              <PressableScale
                onPress={() => handleSwitchSource('screen')}
                style={[
                  styles.sourceOption,
                  currentSource === 'screen' && styles.sourceOptionActive,
                ]}
              >
                <Ionicons
                  name="desktop-outline"
                  size={22}
                  color={currentSource === 'screen' ? palette.accent : palette.textSecondary}
                />
                <Text
                  style={[
                    styles.sourceOptionTitle,
                    currentSource === 'screen' && styles.sourceOptionTitleActive,
                  ]}
                >
                  Desktop Screen
                </Text>
                <Text style={styles.sourceOptionDesc}>Live screen capture</Text>
              </PressableScale>

              <PressableScale
                onPress={() => handleSwitchSource('0')}
                style={[
                  styles.sourceOption,
                  currentSource === '0' && styles.sourceOptionActive,
                ]}
              >
                <Ionicons
                  name="videocam-outline"
                  size={22}
                  color={currentSource === '0' ? palette.accent : palette.textSecondary}
                />
                <Text
                  style={[
                    styles.sourceOptionTitle,
                    currentSource === '0' && styles.sourceOptionTitleActive,
                  ]}
                >
                  Webcam 0
                </Text>
                <Text style={styles.sourceOptionDesc}>USB device index 0</Text>
              </PressableScale>
            </View>

            {/* Custom RTSP / IP Camera Input */}
            <View style={styles.ipCamRow}>
              <TextInput
                value={customUrl}
                onChangeText={setCustomUrl}
                placeholder="IP camera / RTSP url (e.g. http://192.168.1.5:8080/video)"
                placeholderTextColor={palette.textTertiary}
                style={styles.ipInput}
                autoCapitalize="none"
                autoCorrect={false}
              />
              <Button
                variant="secondary"
                label="Connect"
                onPress={handleConnectCustom}
                disabled={switching}
              />
            </View>
          </Card>
        </Animated.View>

        {/* --- Phone Setup & Pair Card --- */}
        <Animated.View entering={enter(3)}>
          <Card padded style={styles.guideCard}>
            <View style={styles.guideHeader}>
              <View style={styles.guideIcon}>
                <Ionicons name="qr-code-outline" size={24} color={palette.cyan} />
              </View>
              <View style={styles.guideMeta}>
                <Text style={styles.guideTitle}>Use Your Phone as a Remote Eye</Text>
                <Text style={styles.guideSubtitle}>
                  Prop your phone up anywhere on your local Wi-Fi to watch rooms or entrances.
                </Text>
              </View>
            </View>

            <View style={styles.urlBox}>
              <Text style={styles.urlLabel}>Open this URL on your phone:</Text>
              <View style={styles.urlRow}>
                <Text style={styles.urlText} numberOfLines={1}>
                  {connectUrl}
                </Text>
                <PressableScale onPress={copyConnectUrl} style={styles.copyBtn}>
                  <Ionicons name="copy-outline" size={16} color={palette.accent} />
                  <Text style={styles.copyBtnText}>Copy</Text>
                </PressableScale>
              </View>
            </View>

            <View style={styles.stepsList}>
              <View style={styles.stepItem}>
                <Text style={styles.stepNum}>1</Text>
                <Text style={styles.stepText}>Connect your phone to the same Wi-Fi network.</Text>
              </View>
              <View style={styles.stepItem}>
                <Text style={styles.stepNum}>2</Text>
                <Text style={styles.stepText}>Open the address above in Safari or Chrome.</Text>
              </View>
              <View style={styles.stepItem}>
                <Text style={styles.stepNum}>3</Text>
                <Text style={styles.stepText}>
                  Tap the <Text style={styles.boldText}>Camera</Text> tab and press{' '}
                  <Text style={styles.boldText}>Broadcast to HawkEye</Text>.
                </Text>
              </View>
            </View>
          </Card>
        </Animated.View>

        <View style={{ height: TAB_BAR_SPACE }} />
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    gap: spacing.lg,
  },
  gear: {
    width: 38,
    height: 38,
    borderRadius: radii.pill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
  },
  viewfinderCard: {
    overflow: 'hidden',
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairlineStrong,
  },
  viewfinderContainer: {
    position: 'relative',
    width: '100%',
    aspectRatio: 16 / 9,
    backgroundColor: '#0a0a0c',
  },
  fallbackBox: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },
  fallbackText: {
    ...typography.body,
    color: palette.textTertiary,
  },
  hudOverlay: {
    ...StyleSheet.absoluteFillObject,
    padding: spacing.md,
    justifyContent: 'space-between',
  },
  hudHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  hudBadgeGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  liveIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: radii.pill,
    backgroundColor: 'rgba(0,0,0,0.65)',
  },
  liveFpsText: {
    ...typography.micro,
    color: palette.green,
    fontWeight: '700',
  },
  hudFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: spacing.sm,
  },
  hudBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: spacing.sm,
    paddingVertical: 6,
    borderRadius: radii.pill,
    backgroundColor: 'rgba(20,20,24,0.75)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.15)',
  },
  hudBtnText: {
    ...typography.caption,
    color: palette.text,
    fontWeight: '600',
  },
  corner: {
    position: 'absolute',
    width: 14,
    height: 14,
    borderColor: palette.cyan,
  },
  cornerTL: { top: 12, left: 12, borderTopWidth: 2, borderLeftWidth: 2 },
  cornerTR: { top: 12, right: 12, borderTopWidth: 2, borderRightWidth: 2 },
  cornerBL: { bottom: 12, left: 12, borderBottomWidth: 2, borderLeftWidth: 2 },
  cornerBR: { bottom: 12, right: 12, borderBottomWidth: 2, borderRightWidth: 2 },
  broadcastActionWrap: {
    padding: spacing.lg,
    gap: spacing.sm,
  },
  broadcastSubtext: {
    ...typography.caption,
    color: palette.textTertiary,
    textAlign: 'center',
  },
  errorNotice: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: palette.redDim,
  },
  errorNoticeText: {
    ...typography.caption,
    color: palette.red,
    flexShrink: 1,
  },
  statRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  statTile: {
    flex: 1,
    padding: spacing.md,
    borderRadius: radii.md,
    backgroundColor: palette.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
    alignItems: 'center',
  },
  statLabel: {
    ...typography.micro,
    color: palette.textTertiary,
  },
  statValue: {
    ...typography.title,
    color: palette.text,
    marginTop: 2,
  },
  sourceCard: {
    gap: spacing.md,
  },
  sectionCaption: {
    ...typography.caption,
    color: palette.textSecondary,
    marginTop: -spacing.xs,
  },
  sourceBtnGrid: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  sourceOption: {
    flex: 1,
    padding: spacing.md,
    borderRadius: radii.md,
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
    alignItems: 'center',
    gap: 4,
  },
  sourceOptionActive: {
    borderColor: palette.accent,
    backgroundColor: palette.accentDim,
  },
  sourceOptionTitle: {
    ...typography.label,
    color: palette.text,
    fontSize: 12,
  },
  sourceOptionTitleActive: {
    color: palette.accent,
    fontWeight: '700',
  },
  sourceOptionDesc: {
    ...typography.micro,
    color: palette.textTertiary,
  },
  ipCamRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    alignItems: 'center',
    marginTop: spacing.xs,
  },
  ipInput: {
    flex: 1,
    ...typography.caption,
    color: palette.text,
    backgroundColor: palette.surfaceElevated,
    borderRadius: radii.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairlineStrong,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  guideCard: {
    gap: spacing.md,
  },
  guideHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  guideIcon: {
    width: 44,
    height: 44,
    borderRadius: radii.md,
    backgroundColor: palette.cyanDim,
    alignItems: 'center',
    justifyContent: 'center',
  },
  guideMeta: {
    flex: 1,
  },
  guideTitle: {
    ...typography.bodyStrong,
    color: palette.text,
  },
  guideSubtitle: {
    ...typography.caption,
    color: palette.textSecondary,
    marginTop: 2,
  },
  urlBox: {
    padding: spacing.md,
    backgroundColor: palette.surfaceElevated,
    borderRadius: radii.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
    gap: spacing.xs,
  },
  urlLabel: {
    ...typography.micro,
    color: palette.textTertiary,
  },
  urlRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.sm,
  },
  urlText: {
    ...typography.bodyStrong,
    color: palette.accent,
    flex: 1,
  },
  copyBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    backgroundColor: palette.accentDim,
    borderRadius: radii.pill,
  },
  copyBtnText: {
    ...typography.caption,
    color: palette.accent,
    fontWeight: '600',
  },
  stepsList: {
    gap: spacing.sm,
  },
  stepItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  stepNum: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: palette.surfaceElevated,
    color: palette.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
    fontSize: 11,
    fontWeight: '700',
  },
  stepText: {
    ...typography.caption,
    color: palette.textSecondary,
    flex: 1,
  },
  boldText: {
    color: palette.text,
    fontWeight: '600',
  },
});
