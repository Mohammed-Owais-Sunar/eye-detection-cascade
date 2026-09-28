# ADPULSE Testing Checklist

Use this checklist before a hackathon demo, LinkedIn recording, CV screenshot or production-like deployment.

## 1. Basic camera test

- [ ] Browser camera permission is granted.
- [ ] Camera starts without a Streamlit error.
- [ ] Video feed is visible.
- [ ] Audio is not required.
- [ ] Camera can be stopped cleanly.

## 2. Single-person engagement

**Expected:** one engagement after approximately 5 seconds of continuous attention.

- [ ] Person enters the camera view.
- [ ] Face is detected.
- [ ] Both eyes are detected when conditions allow.
- [ ] Keep facing the camera for 5+ seconds.
- [ ] Engagement count increases by exactly 1.
- [ ] The same temporary track does not create another engagement while continuing to look.

## 3. Attention interruption

**Expected:** the 5-second timer should reset when attention is lost.

- [ ] Look toward the camera for less than 5 seconds.
- [ ] Look away / hide the eyes.
- [ ] Return to the camera.
- [ ] Confirm the previous partial interval did not become an engagement.

## 4. Multiple people

**Expected:** multiple people can be tracked simultaneously.

- [ ] Test with 2 people.
- [ ] Test with 3 people if the camera view allows it.
- [ ] Confirm separate temporary tracks appear.
- [ ] Have only one person look toward the camera.
- [ ] Confirm engagement is associated with the correct temporary track.
- [ ] Have both people look for 5+ seconds.
- [ ] Confirm both can become engagements.

## 5. Difficult conditions

Run the same test under:

- [ ] Bright indoor lighting
- [ ] Low indoor lighting
- [ ] Backlighting
- [ ] Glasses
- [ ] Slight head rotation
- [ ] Person entering/leaving the frame
- [ ] Two people close together
- [ ] Different webcam resolutions

Record false positives and missed detections.

## 6. Event and campaign flow

- [ ] Create an event.
- [ ] Assign a location.
- [ ] Select/verify the campaign.
- [ ] Activate the event.
- [ ] Start the camera.
- [ ] Generate a test engagement.
- [ ] Verify the event/location/campaign appears in the analytics log.
- [ ] Deactivate the event and verify the UI state changes correctly.

## 7. Cloud deployment

Test the public deployment separately from localhost:

- [ ] Open the Streamlit Cloud URL.
- [ ] Grant camera permission.
- [ ] Start the camera.
- [ ] Repeat the single-person 5-second test.
- [ ] Repeat the 2-person test.
- [ ] Check browser console/app errors if WebRTC gets stuck.
- [ ] Test from at least one additional network if possible.

## 8. Demo evidence

Capture:

- [ ] Live Control Room screenshot
- [ ] Camera with multiple tracks visible
- [ ] Engagement after the 5-second threshold
- [ ] Intelligence Log
- [ ] Event Manager
- [ ] Analytics chart

## Accuracy notes

Do not describe the current system as exact gaze tracking or facial recognition.

Use language such as:

> "Privacy-conscious computer-vision based advertisement attention estimation using face/eye detection and temporary spatial tracking."

The current implementation is affected by camera quality, lighting, glasses, occlusion, face angle and Haar-cascade detection limits.
