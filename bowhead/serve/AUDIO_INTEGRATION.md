"""
Minimal HTML template example for audio streaming.
This shows how the detail panel should request audio from the Flask API.

Place this in your generated HTML's click handler.
"""

html_template = """
<!-- Detail panel audio player (in the onclick handler for each point) -->
<audio id="audio-player" controls style="width: 100%; margin-top: 10px;">
  <source id="audio-source" src="" type="audio/wav">
  Your browser does not support the audio element.
</audio>

<script>
// When user clicks a point on the scatter plot
function showPointDetails(sample_id) {
  const detailPanel = document.getElementById('detail-panel');
  
  // ... existing code to show metadata ...
  
  // Set audio source to Flask API endpoint
  const audioSource = document.getElementById('audio-source');
  audioSource.src = `/api/audio/${sample_id}`;
  
  // Reset player
  const audioPlayer = document.getElementById('audio-player');
  audioPlayer.load();
}
</script>
"""

print(html_template)
