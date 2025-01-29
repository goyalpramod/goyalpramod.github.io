import { getFirestore, doc, getDoc, setDoc, updateDoc, increment } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-firestore.js";

document.addEventListener('DOMContentLoaded', async function() {
    const votingContainer = document.querySelector('.voting-buttons');
    if (!votingContainer) return;
  
    const postId = votingContainer.dataset.postId;
    const upvoteBtn = votingContainer.querySelector('.upvote');
    const downvoteBtn = votingContainer.querySelector('.downvote');
    const upvoteCount = votingContainer.querySelector('.upvote-count');
    const downvoteCount = votingContainer.querySelector('.downvote-count');
  
    // Wait for Firebase to be initialized
    while (!window.firebaseApp) {
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Get Firestore instance
    const db = getFirestore(window.firebaseApp);
    
    // Get votes document reference
    const votesRef = doc(db, 'votes', postId);
  
    // Track voted state
    let hasVoted = false;
  
    // Get initial votes and check local storage
    try {
        const docSnap = await getDoc(votesRef);
        if (!docSnap.exists()) {
            // Initialize with 0 votes using setDoc for new documents
            await setDoc(votesRef, {
                upvotes: 0,
                downvotes: 0
            });
        }
        const data = docSnap.data() || { upvotes: 0, downvotes: 0 };
        upvoteCount.textContent = data.upvotes;
        downvoteCount.textContent = data.downvotes;
        
        // Check if user has voted before
        hasVoted = localStorage.getItem(`voted_${postId}`);
        if (hasVoted === 'up') {
            upvoteBtn.classList.add('voted');
        } else if (hasVoted === 'down') {
            downvoteBtn.classList.add('voted');
        }
    } catch (error) {
        console.error("Error initializing votes:", error);
        // Set default values in UI
        upvoteCount.textContent = '0';
        downvoteCount.textContent = '0';
    }
  
    // Handle upvote
    upvoteBtn.addEventListener('click', async () => {
        if (hasVoted) return; // Prevent multiple votes
        
        try {
            await updateDoc(votesRef, {
                upvotes: increment(1)
            });
            
            const updated = await getDoc(votesRef);
            upvoteCount.textContent = updated.data().upvotes;
            
            // Mark as voted
            localStorage.setItem(`voted_${postId}`, 'up');
            upvoteBtn.classList.add('voted');
            hasVoted = 'up';
        } catch (error) {
            console.error("Error upvoting:", error);
        }
    });
  
    // Handle downvote
    downvoteBtn.addEventListener('click', async () => {
        if (hasVoted) return; // Prevent multiple votes
        
        try {
            await updateDoc(votesRef, {
                downvotes: increment(1)
            });
            
            const updated = await getDoc(votesRef);
            downvoteCount.textContent = updated.data().downvotes;
            
            // Mark as voted
            localStorage.setItem(`voted_${postId}`, 'down');
            downvoteBtn.classList.add('voted');
            hasVoted = 'down';
        } catch (error) {
            console.error("Error downvoting:", error);
        }
    });
});