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
    let currentData = { upvotes: 0, downvotes: 0 };
  
    // Initialize or get votes
    try {
        const docSnap = await getDoc(votesRef);
        if (docSnap.exists()) {
            currentData = docSnap.data();
        } else {
            // Create the document if it doesn't exist
            try {
                await setDoc(votesRef, currentData);
            } catch (error) {
                console.error("Error creating vote document:", error);
            }
        }
        
        // Update UI
        upvoteCount.textContent = currentData.upvotes;
        downvoteCount.textContent = currentData.downvotes;
        
        // Check if user has voted before
        hasVoted = localStorage.getItem(`voted_${postId}`);
        if (hasVoted === 'up') {
            upvoteBtn.classList.add('voted');
        } else if (hasVoted === 'down') {
            downvoteBtn.classList.add('voted');
        }
    } catch (error) {
        console.error("Error initializing votes:", error);
        // Keep default values in UI
        upvoteCount.textContent = '0';
        downvoteCount.textContent = '0';
    }
  
    // Handle upvote
    upvoteBtn.addEventListener('click', async () => {
        if (hasVoted) return;
        
        try {
            const newUpvotes = currentData.upvotes + 1;
            await setDoc(votesRef, {
                ...currentData,
                upvotes: newUpvotes
            });
            
            currentData.upvotes = newUpvotes;
            upvoteCount.textContent = newUpvotes;
            
            localStorage.setItem(`voted_${postId}`, 'up');
            upvoteBtn.classList.add('voted');
            hasVoted = 'up';
        } catch (error) {
            console.error("Error upvoting:", error);
        }
    });
  
    // Handle downvote
    downvoteBtn.addEventListener('click', async () => {
        if (hasVoted) return;
        
        try {
            const newDownvotes = currentData.downvotes + 1;
            await setDoc(votesRef, {
                ...currentData,
                downvotes: newDownvotes
            });
            
            currentData.downvotes = newDownvotes;
            downvoteCount.textContent = newDownvotes;
            
            localStorage.setItem(`voted_${postId}`, 'down');
            downvoteBtn.classList.add('voted');
            hasVoted = 'down';
        } catch (error) {
            console.error("Error downvoting:", error);
        }
    });
});